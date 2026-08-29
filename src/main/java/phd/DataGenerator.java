package phd;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.*;
import net.datafaker.Faker;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.stream.Collectors;

/**
 * The {@code DataGenerator} class is responsible for instantiating synthetic data
 * from JSON templates and compiling a labeled dataset suitable for training or evaluating
 * Machine Learning models (specifically for PII detection/classification).
 * * <p><b>Core Objective:</b></p>
 * To produce a strictly balanced dataset (e.g., 5000 records with a 70% PII / 30% Non-PII ratio)
 * where every payload is accompanied by ground-truth labels indicating exactly where
 * and what type of PII is present.
 *
 * <p><b>Main Execution Pipeline:</b></p>
 * <ol>
 * <li><b>Template Loading:</b> Reads all JSON template variations generated previously from the {@code generated_templates} directory.</li>
 * <li><b>Rejection Sampling Loop:</b> Randomly selects a template and populates it with fake data. It then evaluates if the resulting payload contains PII. The record is only added to the final dataset if its respective quota (targetPiiCount or targetNonPiiCount) is not yet fulfilled. This guarantees exact dataset ratios.</li>
 * <li><b>Shuffling & Output:</b> The final dataset is shuffled to prevent ordering bias and written to a JSON Lines ({@code .jsonl}) file. A detailed analytical text report is also generated.</li>
 * </ol>
 *
 * <p><b>Data Instantiation ({@code fillTemplate} & {@code replaceWithFakeData}):</b></p>
 * The algorithm recursively traverses the JSON tree and resolves placeholders:
 * <ul>
 * <li><b>PII Generation:</b> Detects text values starting with {@code "PII_"}. Uses the {@code net.datafaker.Faker} library to inject highly realistic mock data (e.g., replacing {@code PII_EMAIL} with a realistic email address).</li>
 * <li><b>Entity Tracking:</b> Whenever a PII value is generated, the algorithm records its exact JSON Path, the PII type, and the generated value into an {@code entities} array for ground-truth labeling.</li>
 * <li><b>System Types:</b> Detects standard system markers (e.g., {@code "STRING"}, {@code "INT"}, {@code "DATETIME"}) and replaces them with safe, random, non-PII values.</li>
 * <li><b>Dynamic Arrays:</b> If an array contains exactly one template element, the algorithm treats it as an "expandable schema" and dynamically generates between 1 and 4 instances of that element.</li>
 * </ul>
 *
 * <p><b>Ground-Truth Wrapper ({@code wrapRecord}):</b></p>
 * Transforms the raw populated JSON into a supervised learning format. The final output structure looks like:
 * <pre>
 * {
 * "payload": { ... the actual data ... },
 * "labels": {
 * "contains_pii": true,
 * "entities": [
 * { "type": "PII_EMAIL", "value": "john.doe@example.com", "json_path": "$.user.contact.email" }
 * ]
 * }
 * }
 * </pre>
 */
public class DataGenerator {

    private static final int TOTAL_RECORDS = 20000;
    private static final double PII_RATIO = 0.70;
    private static final ObjectMapper mapper = new ObjectMapper();
    private static final Faker faker = new Faker(new Locale("en-US"));
    private static final Random random = new Random();
    private static final JsonNodeFactory factory = JsonNodeFactory.instance;
    private static final String OUTPUT_FILE = "generated_data/dataset_with_labels.jsonl";
    private static final String OUTPUT_TRAINING_FILE = "generated_data/dataset_training_5k_v3.jsonl";
    private static final String OUTPUT_REPORT_FILE = "generated_data/dataset_report.txt";

    // Training subset
    private static final int    TRAINING_SUBSET_SIZE = 5000;
    private static final double TRAINING_PII_RATIO   = 0.65; // 65% PII / 35% non-PII
    private static final int    TRAINING_MIN_PER_CLASS = 100; // min records per dominant PII class
    private static final String INPUT_DIR = "generated_templates";
    private static final List<String> SYSTEM_TYPES = Arrays.asList("STRING", "INT", "DECIMAL", "BOOLEAN", "DATE", "DATETIME", "DATETIME_EPOCH", "URL", "OBJECT", "ARRAY", "NULL", "TEXT");

    // Minimum share (fraction of total PII entities) each PII type must reach.
    // Types not listed here have no enforced minimum and fill the remainder naturally.
    private static final Map<String, Double> PII_TYPE_MIN_SHARE = new LinkedHashMap<>();
    static {
        PII_TYPE_MIN_SHARE.put("PII_PERSON_NAME",        0.08);
        PII_TYPE_MIN_SHARE.put("PII_PERSON_FIRST_NAME",  0.05);
        PII_TYPE_MIN_SHARE.put("PII_PERSON_LAST_NAME",   0.04);
        PII_TYPE_MIN_SHARE.put("PII_EMAIL",              0.10);
        PII_TYPE_MIN_SHARE.put("PII_PHONE_NUMBER",       0.08);
        PII_TYPE_MIN_SHARE.put("PII_ADDRESS_LINE",       0.07);
        PII_TYPE_MIN_SHARE.put("PII_LOCATION_CITY",      0.06);
        PII_TYPE_MIN_SHARE.put("PII_LOCATION_COUNTRY",   0.06);
        PII_TYPE_MIN_SHARE.put("PII_DATE_OF_BIRTH",      0.02);
        PII_TYPE_MIN_SHARE.put("PII_NATIONAL_ID",        0.015);
        PII_TYPE_MIN_SHARE.put("PII_IBAN",               0.02);
        PII_TYPE_MIN_SHARE.put("PII_COMPANY_NAME",       0.05);
        PII_TYPE_MIN_SHARE.put("PII_USERNAME",           0.02);
        PII_TYPE_MIN_SHARE.put("PII_PASSWORD",           0.01);
        PII_TYPE_MIN_SHARE.put("PII_ORGANIZATION_NAME",  0.025);
        PII_TYPE_MIN_SHARE.put("PII_LOCATION_STATE",     0.03);
        PII_TYPE_MIN_SHARE.put("PII_IP_ADDRESS",         0.025);
        PII_TYPE_MIN_SHARE.put("PII_MAC_ADDRESS",        0.01);
        PII_TYPE_MIN_SHARE.put("PII_CREDIT_CARD",        0.01);
    }

    // ISO-8601 Formatter for genuine random dates
    private static final DateTimeFormatter ISO_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss'Z'").withZone(ZoneId.of("UTC"));

    public static void main(String[] args) throws Exception {
        Path inputPath = Paths.get(INPUT_DIR);
        Path outputPath = Paths.get("generated_data");

        if (!Files.exists(outputPath)) Files.createDirectories(outputPath);

        List<JsonNode> allTemplates = new ArrayList<>();
        File[] files = inputPath.toFile().listFiles((dir, name) -> name.endsWith(".json"));
        if (files == null || files.length == 0) return;

        for (File f : files) {
            allTemplates.add(mapper.readTree(f));
        }

        int targetPiiCount = (int) (TOTAL_RECORDS * PII_RATIO);
        int targetNonPiiCount = TOTAL_RECORDS - targetPiiCount;

        int generatedPii = 0;
        int generatedNonPii = 0;

        List<String> finalDataset = new ArrayList<>(TOTAL_RECORDS);

        // Map to track the distribution of PII types
        Map<String, Integer> piiDistribution = new HashMap<>();

        // Separate template lists for targeted second-pass generation
        List<JsonNode> piiTemplates = allTemplates.stream()
                .filter(t -> containsPiiMarker(t))
                .collect(Collectors.toList());

        System.out.println("Starting dataset generation. Target PII: " + targetPiiCount + ", Target Non-PII: " + targetNonPiiCount);
        System.out.println("Total templates: " + allTemplates.size() + " (PII: " + piiTemplates.size() + ")");

        // PASS 0 — PLAIN-TEXT RECORDS (ai4privacy-style prose, no JSON wrapper).
        // Evaluation traffic includes bare free-text messages; without these the model
        // only ever sees prose inside JSON fields and generalizes poorly to plain text.
        int plainPiiTarget    = (int) (TOTAL_RECORDS * 0.06); // prose WITH embedded PII
        int plainNonPiiTarget = (int) (TOTAL_RECORDS * 0.06); // neutral prose incl. hard negatives (v3: raised from 4%)

        for (int i = 0; i < plainPiiTarget && generatedPii < targetPiiCount; i++) {
            ArrayNode entities = factory.arrayNode();
            JsonNode payload = plainTextPayload(true, entities);
            finalDataset.add(wrapRecord(payload, entities, true));
            generatedPii++;
            for (JsonNode entity : entities) {
                piiDistribution.merge(entity.get("type").asText(), 1, Integer::sum);
            }
        }
        for (int i = 0; i < plainNonPiiTarget && generatedNonPii < targetNonPiiCount; i++) {
            ArrayNode entities = factory.arrayNode();
            JsonNode payload = plainTextPayload(false, entities);
            finalDataset.add(wrapRecord(payload, entities, false));
            generatedNonPii++;
        }
        System.out.println("Pass 0 complete: " + generatedPii + " plain-text PII + " + generatedNonPii + " plain-text non-PII records.");

        // PASS 1 — REJECTION SAMPLING LOOP: Guarantees ground-truth labels and strict dataset ratios
        while (generatedPii < targetPiiCount || generatedNonPii < targetNonPiiCount) {
            JsonNode template = allTemplates.get(random.nextInt(allTemplates.size()));
            ArrayNode entities = factory.arrayNode();

            JsonNode filledPayload = fillTemplate(template.deepCopy(), "$", entities);
            boolean isActuallyPii = entities.size() > 0;

            if (isActuallyPii && generatedPii < targetPiiCount) {
                finalDataset.add(wrapRecord(filledPayload, entities, true));
                generatedPii++;
                for (JsonNode entity : entities) {
                    String type = entity.get("type").asText();
                    piiDistribution.put(type, piiDistribution.getOrDefault(type, 0) + 1);
                }
            } else if (!isActuallyPii && generatedNonPii < targetNonPiiCount) {
                finalDataset.add(wrapRecord(filledPayload, entities, false));
                generatedNonPii++;
            }
        }

        // PASS 2 — BALANCE PII TYPE DISTRIBUTION
        // For each PII type below its minimum share, replace the lowest-frequency PII record
        // in the dataset with a freshly generated record guaranteed to contain that type.
        System.out.println("Pass 1 complete. Running PII distribution balancing pass...");
        int totalEntitiesPass1 = piiDistribution.values().stream().mapToInt(Integer::intValue).sum();

        for (Map.Entry<String, Double> entry : PII_TYPE_MIN_SHARE.entrySet()) {
            String targetType = entry.getKey();
            int minCount = (int) (totalEntitiesPass1 * entry.getValue());
            int currentCount = piiDistribution.getOrDefault(targetType, 0);

            if (currentCount >= minCount) continue;

            // Find templates that contain this specific PII type
            List<JsonNode> typedTemplates = piiTemplates.stream()
                    .filter(t -> templateContainsType(t, targetType))
                    .collect(Collectors.toList());

            if (typedTemplates.isEmpty()) {
                System.out.println("Warning: no template found for type " + targetType);
                continue;
            }

            int needed = minCount - currentCount;
            int added = 0;
            int maxAttempts = needed * 50;
            int attempts = 0;

            while (added < needed && attempts < maxAttempts) {
                attempts++;
                JsonNode template = typedTemplates.get(random.nextInt(typedTemplates.size()));
                ArrayNode entities = factory.arrayNode();
                JsonNode filledPayload = fillTemplate(template.deepCopy(), "$", entities);

                boolean hasTargetType = false;
                for (JsonNode e : entities) {
                    if (e.get("type").asText().equals(targetType)) { hasTargetType = true; break; }
                }
                if (!hasTargetType) continue;

                // Replace a random existing PII record in the dataset (keeps total count stable).
                // Must never replace a non-PII record (ratio would drift) nor a plain-text
                // record (their share is a deliberate design target).
                int replaceIdx;
                JsonNode oldRecord;
                boolean replaceable;
                int idxAttempts = 0;
                do {
                    replaceIdx = random.nextInt(finalDataset.size());
                    oldRecord = mapper.readTree(finalDataset.get(replaceIdx));
                    replaceable = oldRecord.path("labels").path("contains_pii").asBoolean(false)
                            && !oldRecord.path("payload").isTextual();
                    idxAttempts++;
                } while (!replaceable && idxAttempts < 100);
                if (!replaceable) continue;

                // Remove old record's entities from distribution before replacing
                JsonNode oldEntities = oldRecord.path("labels").path("entities");
                if (oldEntities.isArray()) {
                    for (JsonNode e : oldEntities) {
                        String oldType = e.get("type").asText();
                        piiDistribution.merge(oldType, -1, Integer::sum);
                    }
                }
                finalDataset.set(replaceIdx, wrapRecord(filledPayload, entities, true));

                for (JsonNode entity : entities) {
                    String type = entity.get("type").asText();
                    piiDistribution.merge(type, 1, Integer::sum);
                }
                added++;
            }
            System.out.printf("  %-30s deficit=%d  added=%d%n", targetType, needed, added);
        }

        Collections.shuffle(finalDataset);

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(OUTPUT_FILE))) {
            for (String record : finalDataset) {
                writer.write(record);
                writer.newLine();
            }
        }

        System.out.println("Dataset generation complete! Strict ratio enforced: " + OUTPUT_FILE);

        // PASS 3 — TRAINING SUBSET
        writeTrainingSubset(finalDataset);

        // Print the detailed analytical report
        printReport(generatedPii, generatedNonPii, piiDistribution);
    }

    /**
     * Writes a stratified training subset: TRAINING_PII_RATIO of PII records
     * (guaranteeing TRAINING_MIN_PER_CLASS per dominant PII class) plus non-PII
     * records for the remainder. Negatives are essential - a model trained only
     * on PII-bearing records never learns to answer "no PII".
     */
    private static void writeTrainingSubset(List<String> fullDataset) throws Exception {
        Map<String, List<String>> byDominantType = new HashMap<>();
        List<String> piiRecords = new ArrayList<>();
        List<String> nonPiiRecords = new ArrayList<>();

        for (String rec : fullDataset) {
            JsonNode entities = mapper.readTree(rec).path("labels").path("entities");
            if (entities.isArray() && entities.size() > 0) {
                piiRecords.add(rec);
                Map<String, Integer> counts = new HashMap<>();
                for (JsonNode e : entities) {
                    counts.merge(e.get("type").asText(), 1, Integer::sum);
                }
                String dominant = counts.entrySet().stream()
                        .max(Map.Entry.comparingByValue()).get().getKey();
                byDominantType.computeIfAbsent(dominant, k -> new ArrayList<>()).add(rec);
            } else {
                nonPiiRecords.add(rec);
            }
        }

        int targetPii = (int) (TRAINING_SUBSET_SIZE * TRAINING_PII_RATIO);

        // Phase 1: guaranteed minimum per dominant class
        LinkedHashSet<String> chosenPii = new LinkedHashSet<>();
        for (List<String> group : byDominantType.values()) {
            Collections.shuffle(group, random);
            chosenPii.addAll(group.subList(0, Math.min(TRAINING_MIN_PER_CLASS, group.size())));
        }

        // Phase 2: top up proportionally to the PII target
        Collections.shuffle(piiRecords, random);
        for (String rec : piiRecords) {
            if (chosenPii.size() >= targetPii) break;
            chosenPii.add(rec);
        }

        // Phase 3: negatives fill the remainder
        List<String> subset = new ArrayList<>(chosenPii);
        Collections.shuffle(nonPiiRecords, random);
        int negCount = Math.min(TRAINING_SUBSET_SIZE - subset.size(), nonPiiRecords.size());
        subset.addAll(nonPiiRecords.subList(0, negCount));

        Collections.shuffle(subset, random);

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(OUTPUT_TRAINING_FILE))) {
            for (String record : subset) {
                writer.write(record);
                writer.newLine();
            }
        }
        System.out.printf("Training subset written: %s (%d records: %d PII / %d non-PII)%n",
                OUTPUT_TRAINING_FILE, subset.size(), chosenPii.size(), negCount);
    }

    /**
     * Prints a formatted report regarding the dataset composition and PII distribution.
     */
    private static void printReport(int generatedPii, int generatedNonPii, Map<String, Integer> piiDistribution) {
        try (PrintWriter out = new PrintWriter(new FileOutputStream(OUTPUT_REPORT_FILE))){
            out.println("==================================================");
            out.println("DATASET DISTRIBUTION REPORT");
            out.println("==================================================");
            out.println("Total JSONL records   : " + (generatedPii + generatedNonPii));
            out.println("PII records count     : " + generatedPii + " (" + String.format("%.0f", PII_RATIO * 100) + "%)");
            out.println("Non-PII records count : " + generatedNonPii + " (" + String.format("%.0f", (1 - PII_RATIO) * 100) + "%)");

            // Calculate the TOTAL number of individual PII entities generated across all records
            int totalPiiEntities = piiDistribution.values().stream().mapToInt(Integer::intValue).sum();

            out.println("Total PII entities    : " + totalPiiEntities + " (individual fields)");
            out.println("--------------------------------------------------");
            out.println("Detailed breakdown of PII classes (frequency & %):");

            // Sort by frequency (descending) and calculate relative percentages
            piiDistribution.entrySet().stream()
                    .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                    .forEach(entry -> {
                        double percentage = (entry.getValue() * 100.0) / totalPiiEntities;
                        out.printf(" - %-25s : %-5d (%.2f%%)%n", entry.getKey(), entry.getValue(), percentage);
                    });
            out.println("==================================================");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    private static String wrapRecord(JsonNode payload, ArrayNode entities, boolean isPii) throws Exception {
        ObjectNode labels = factory.objectNode();
        boolean isActuallyPii = entities.size() > 0;
        labels.put("contains_pii", isActuallyPii);

        // Sanity check
        if(isActuallyPii != isPii) {
            System.err.println("CRITICAL ERROR: Status mismatch for payload: " + payload);
        }

        labels.set("entities", entities);

        ObjectNode root = factory.objectNode();
        root.set("payload", payload);
        root.set("labels", labels);

        return mapper.writeValueAsString(root);
    }

    private static JsonNode fillTemplate(JsonNode node, String currentPath, ArrayNode entities) {
        if (node.isObject()) {
            ObjectNode obj = (ObjectNode) node;
            List<String> fieldNames = new ArrayList<>();
            obj.fieldNames().forEachRemaining(fieldNames::add);

            for (String fieldName : fieldNames) {
                JsonNode child = obj.get(fieldName);
                String newPath = currentPath + "." + fieldName;
                JsonNode updatedChild = fillTemplate(child, newPath, entities);
                obj.set(fieldName, updatedChild);
            }
            return obj;

        } else if (node.isArray()) {
            ArrayNode arr = (ArrayNode) node;
            ArrayNode newArr = factory.arrayNode();

            // Expand dynamic arrays or keep static structures
            if (isExpandableArraySchema(arr)) {
                JsonNode itemTemplate = arr.get(0);
                int randomSize = 1 + random.nextInt(2); // Generates 1 to 2 elements
                for (int i = 0; i < randomSize; i++) {
                    String newPath = currentPath + "[" + i + "]";
                    newArr.add(fillTemplate(itemTemplate.deepCopy(), newPath, entities));
                }
            } else {
                for (int i = 0; i < arr.size(); i++) {
                    String newPath = currentPath + "[" + i + "]";
                    newArr.add(fillTemplate(arr.get(i), newPath, entities));
                }
            }
            return newArr;

        } else if (node.isTextual()) {
            return replaceWithFakeData(node.asText(), currentPath, entities);
        }

        return node;
    }

    /**
     * Helper method to validate if an array acts as an expandable schema.
     */
    private static boolean isExpandableArraySchema(ArrayNode arr) {
        if (arr.size() != 1) return false;
        JsonNode child = arr.get(0);
        return child.isObject() || (child.isTextual() && (child.asText().startsWith("PII_") || isSystemMarker(child.asText())));
    }

    private static boolean isSystemMarker(String val) {
        return SYSTEM_TYPES.contains(val);
    }

    private static boolean containsPiiMarker(JsonNode node) {
        if (node.isObject()) {
            Iterator<JsonNode> it = node.elements();
            while (it.hasNext()) { if (containsPiiMarker(it.next())) return true; }
        } else if (node.isArray()) {
            for (JsonNode item : node) { if (containsPiiMarker(item)) return true; }
        } else if (node.isTextual()) {
            return node.asText().startsWith("PII_");
        }
        return false;
    }

    private static boolean templateContainsType(JsonNode node, String piiType) {
        if (node.isObject()) {
            Iterator<JsonNode> it = node.elements();
            while (it.hasNext()) { if (templateContainsType(it.next(), piiType)) return true; }
        } else if (node.isArray()) {
            for (JsonNode item : node) { if (templateContainsType(item, piiType)) return true; }
        } else if (node.isTextual()) {
            return node.asText().equals(piiType);
        }
        return false;
    }

    private static final Map<String, String[]> FIELD_VALUE_MAP = new HashMap<>();
    static {
        FIELD_VALUE_MAP.put("status",        new String[]{"active", "inactive", "pending", "suspended", "cancelled", "completed", "failed", "processing", "draft", "archived"});
        FIELD_VALUE_MAP.put("state",         new String[]{"active", "inactive", "pending", "suspended", "cancelled", "completed", "failed", "open", "closed", "resolved"});
        FIELD_VALUE_MAP.put("type",          new String[]{"standard", "premium", "basic", "enterprise", "trial", "internal", "external", "manual", "automated", "scheduled"});
        FIELD_VALUE_MAP.put("role",          new String[]{"admin", "user", "viewer", "editor", "owner", "member", "guest", "operator", "auditor", "developer"});
        FIELD_VALUE_MAP.put("currency",      new String[]{"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "SEK", "NOK", "DKK"});
        FIELD_VALUE_MAP.put("environment",   new String[]{"production", "staging", "development", "testing", "sandbox", "qa", "uat", "preview"});
        FIELD_VALUE_MAP.put("language",      new String[]{"en", "de", "fr", "es", "it", "pt", "nl", "pl", "sv", "ja"});
        FIELD_VALUE_MAP.put("severity",      new String[]{"critical", "high", "medium", "low", "info", "warning", "debug", "error"});
        FIELD_VALUE_MAP.put("priority",      new String[]{"urgent", "high", "medium", "low", "normal", "critical"});
        FIELD_VALUE_MAP.put("level",         new String[]{"ERROR", "WARN", "INFO", "DEBUG", "TRACE"});
        FIELD_VALUE_MAP.put("method",        new String[]{"GET", "POST", "PUT", "PATCH", "DELETE"});
        FIELD_VALUE_MAP.put("scheme",        new String[]{"VISA", "MASTERCARD", "AMEX", "DISCOVER", "JCB", "UNIONPAY"});
        FIELD_VALUE_MAP.put("gender",        new String[]{"male", "female", "non-binary", "prefer_not_to_say"});
        FIELD_VALUE_MAP.put("industry",      new String[]{"Technology", "Healthcare", "Finance", "Retail", "Manufacturing", "Education", "Logistics", "Media", "Government", "Energy"});
        FIELD_VALUE_MAP.put("plan",          new String[]{"starter", "basic", "professional", "enterprise", "free", "business", "unlimited"});
        FIELD_VALUE_MAP.put("category",      new String[]{"billing", "technical", "account", "security", "general", "sales", "compliance", "onboarding"});
        FIELD_VALUE_MAP.put("region",        new String[]{"us-east-1", "eu-west-1", "ap-southeast-1", "us-west-2", "eu-central-1", "ap-northeast-1"});
        FIELD_VALUE_MAP.put("datacenter",    new String[]{"us-east-1", "eu-west-1", "ap-southeast-1", "us-west-2", "eu-central-1"});
        FIELD_VALUE_MAP.put("timezone",      new String[]{"UTC", "America/New_York", "Europe/London", "Europe/Berlin", "Asia/Tokyo", "Australia/Sydney"});
        FIELD_VALUE_MAP.put("format",        new String[]{"json", "xml", "csv", "parquet", "avro", "protobuf"});
        FIELD_VALUE_MAP.put("protocol",      new String[]{"https", "http", "grpc", "amqp", "mqtt", "websocket"});
        FIELD_VALUE_MAP.put("outcome",       new String[]{"success", "failure", "partial", "timeout", "skipped"});
        FIELD_VALUE_MAP.put("result",        new String[]{"pass", "fail", "error", "skipped", "pending", "inconclusive"});
        FIELD_VALUE_MAP.put("direction",     new String[]{"inbound", "outbound", "internal"});
        FIELD_VALUE_MAP.put("channel",       new String[]{"email", "sms", "push", "webhook", "slack", "in-app"});
        FIELD_VALUE_MAP.put("source",        new String[]{"api", "webhook", "manual", "import", "sync", "migration", "batch"});
        FIELD_VALUE_MAP.put("trigger",       new String[]{"manual", "scheduled", "webhook", "event", "api", "ci"});
        FIELD_VALUE_MAP.put("strategy",      new String[]{"rolling", "blue-green", "canary", "recreate", "a-b-test"});
        FIELD_VALUE_MAP.put("framework",     new String[]{"pytorch", "tensorflow", "sklearn", "xgboost", "keras", "onnx"});
        FIELD_VALUE_MAP.put("legalBasis",    new String[]{"consent", "legitimate_interest", "contract", "legal_obligation", "vital_interests", "public_task"});
        FIELD_VALUE_MAP.put("grantType",     new String[]{"authorization_code", "client_credentials", "refresh_token", "password", "implicit"});
        FIELD_VALUE_MAP.put("claimType",     new String[]{"medical", "dental", "vision", "auto", "property", "liability", "life"});
        FIELD_VALUE_MAP.put("eventType",     new String[]{"user.created", "user.updated", "user.deleted", "payment.succeeded", "payment.failed", "order.placed", "session.started"});
        FIELD_VALUE_MAP.put("department",    new String[]{"Engineering", "Sales", "Marketing", "Finance", "HR", "Legal", "Operations", "Product", "Support", "Security"});
        FIELD_VALUE_MAP.put("title",         new String[]{"Software Engineer", "Product Manager", "Data Scientist", "DevOps Engineer", "Account Executive", "HR Manager", "CTO", "CFO"});
        FIELD_VALUE_MAP.put("relationship",  new String[]{"spouse", "parent", "sibling", "child", "partner", "friend", "colleague"});
        FIELD_VALUE_MAP.put("bedType",       new String[]{"single", "double", "queen", "king", "twin", "suite"});
        FIELD_VALUE_MAP.put("cabin",         new String[]{"economy", "premium_economy", "business", "first"});
        FIELD_VALUE_MAP.put("sentiment",     new String[]{"positive", "neutral", "negative", "mixed"});
        FIELD_VALUE_MAP.put("intent",        new String[]{"refund", "support", "inquiry", "complaint", "cancellation", "upgrade"});
        FIELD_VALUE_MAP.put("nationality",   new String[]{"American", "German", "French", "British", "Japanese", "Brazilian", "Canadian", "Australian", "Indian", "Dutch"});
        FIELD_VALUE_MAP.put("mfaMethod",     new String[]{"totp", "sms", "email", "hardware_key", "push_notification"});
        FIELD_VALUE_MAP.put("platform",      new String[]{"ios", "android", "web", "windows", "macos", "linux"});
        FIELD_VALUE_MAP.put("branch",        new String[]{"main", "master", "develop", "release/1.0", "feature/auth", "hotfix/login"});
        FIELD_VALUE_MAP.put("purpose",       new String[]{"home_purchase", "debt_consolidation", "business", "education", "auto", "personal", "renovation"});
        FIELD_VALUE_MAP.put("chargesBearer", new String[]{"DEBT", "CRED", "SHAR", "SLEV"});
        FIELD_VALUE_MAP.put("encounterType", new String[]{"outpatient", "inpatient", "emergency", "virtual", "follow-up", "routine"});
        FIELD_VALUE_MAP.put("urgency",       new String[]{"routine", "urgent", "asap", "stat"});
        FIELD_VALUE_MAP.put("deliveredVia",  new String[]{"email", "secure_portal", "postal", "api"});
        FIELD_VALUE_MAP.put("dataResidency", new String[]{"EU", "US", "APAC", "UK", "CA"});
        FIELD_VALUE_MAP.put("object",        new String[]{"fine_tuning.job", "model", "event", "file", "subscription", "customer", "payment"});
        FIELD_VALUE_MAP.put("model",         new String[]{"gpt-4o", "claude-opus-4-8", "gemini-1.5-pro", "llama-3.1-70b", "mistral-large"});
        FIELD_VALUE_MAP.put("stopReason",    new String[]{"end_turn", "max_tokens", "stop_sequence", "tool_use"});
        FIELD_VALUE_MAP.put("employmentType",new String[]{"full-time", "part-time", "contractor", "intern", "freelance"});
        FIELD_VALUE_MAP.put("payFrequency",  new String[]{"weekly", "bi-weekly", "monthly", "semi-monthly"});
        FIELD_VALUE_MAP.put("enrollmentStatus", new String[]{"enrolled", "graduated", "suspended", "withdrawn", "deferred"});
        FIELD_VALUE_MAP.put("applicationStatus", new String[]{"applied", "screening", "interviewing", "offer", "hired", "rejected"});
        FIELD_VALUE_MAP.put("checkType",     new String[]{"document", "facial", "liveness", "database", "aml", "sanctions"});
    }

    private static String realisticString(String currentPath) {
        // Extract the last field name from the JSON path (e.g. "$.user.status" -> "status")
        String fieldName = currentPath.contains(".")
                ? currentPath.substring(currentPath.lastIndexOf('.') + 1)
                : currentPath;
        // Strip array indices like "[0]"
        fieldName = fieldName.replaceAll("\\[\\d+]", "").toLowerCase();

        String[] values = FIELD_VALUE_MAP.get(fieldName);
        if (values != null) {
            return values[random.nextInt(values.length)];
        }

        // Fallback for unrecognised field names
        int r = random.nextInt(4);
        return switch (r) {
            case 0 -> faker.lorem().word();
            case 1 -> faker.numerify("########");
            case 2 -> faker.internet().slug();
            default -> faker.lorem().sentence(2, 2);
        };
    }

    // =====================================================================================
    // FREE-TEXT SUPPORT
    //
    // Real API payloads carry prose fields (ticket descriptions, email bodies, chat
    // messages) where PII is embedded inside sentences rather than isolated in typed
    // fields. Models trained only on field-level PII fail on such text. Two markers:
    //   PII_FREE_TEXT - prose with 1-3 embedded PII values, each recorded as an entity
    //   TEXT          - realistic business/system prose with NO PII (hard negative)
    // =====================================================================================

    /** Sentence templates for PII-bearing prose. Placeholders map to PII types below. */
    private static final String[] PII_TEXT_TEMPLATES = {
        "Hi {PERSON_NAME}, thanks for reaching out. We will reply to {EMAIL} within 24 hours.",
        "Customer {PERSON_NAME} called from {PHONE_NUMBER} regarding a delayed shipment.",
        "Please forward the contract to {PERSON_NAME} at {EMAIL} before the end of the week.",
        "Order confirmed. Shipping to {ADDRESS_LINE}, {LOCATION_CITY}. Contact: {PHONE_NUMBER}.",
        "Meeting scheduled with {PERSON_NAME} from {COMPANY_NAME} to discuss the renewal terms.",
        "User {USERNAME} reported a login failure from IP {IP_ADDRESS} at 09:41 UTC.",
        "Refund of the last invoice has been issued to {PERSON_NAME}. A confirmation was sent to {EMAIL}.",
        "Dear {PERSON_NAME}, your appointment has been rescheduled. If this does not suit you, call us at {PHONE_NUMBER}.",
        "Escalating to tier 2: account owner {PERSON_NAME} ({EMAIL}) cannot reset the password.",
        "New lead created: {PERSON_NAME}, {COMPANY_NAME}, {LOCATION_CITY}. Phone {PHONE_NUMBER}.",
        "Package could not be delivered to {ADDRESS_LINE}. Recipient {PERSON_NAME} was not available.",
        "I spoke with {PERSON_NAME} yesterday, she asked us to update the billing email to {EMAIL}.",
        "Interview feedback for candidate {PERSON_NAME}: strong communication, available from June. CV sent from {EMAIL}.",
        "Payment received from {PERSON_NAME}, IBAN {IBAN}. Please reconcile with invoice 4471.",
        "Suspicious activity: card ending in digits of {CREDIT_CARD} used by {PERSON_NAME} from {LOCATION_CITY}.",
        "Support session started by agent for user {USERNAME} ({EMAIL}). Device IP: {IP_ADDRESS}.",
        "Note: {PERSON_NAME} moved to {ADDRESS_LINE}, {LOCATION_CITY}. Update the delivery preferences.",
        "Reminder sent to {EMAIL} about the upcoming subscription renewal.",
        "Driver {PERSON_NAME} will pick up the package. You can reach the driver at {PHONE_NUMBER}.",
        "Complaint filed by {PERSON_NAME} ({EMAIL}): the invoice lists the wrong company, should be {COMPANY_NAME}."
    };

    private static final Map<String, String> TEXT_PLACEHOLDER_TYPES = new LinkedHashMap<>();
    static {
        TEXT_PLACEHOLDER_TYPES.put("{PERSON_NAME}",   "PII_PERSON_NAME");
        TEXT_PLACEHOLDER_TYPES.put("{EMAIL}",         "PII_EMAIL");
        TEXT_PLACEHOLDER_TYPES.put("{PHONE_NUMBER}",  "PII_PHONE_NUMBER");
        TEXT_PLACEHOLDER_TYPES.put("{ADDRESS_LINE}",  "PII_ADDRESS_LINE");
        TEXT_PLACEHOLDER_TYPES.put("{LOCATION_CITY}", "PII_LOCATION_CITY");
        TEXT_PLACEHOLDER_TYPES.put("{COMPANY_NAME}",  "PII_COMPANY_NAME");
        TEXT_PLACEHOLDER_TYPES.put("{USERNAME}",      "PII_USERNAME");
        TEXT_PLACEHOLDER_TYPES.put("{IP_ADDRESS}",    "PII_IP_ADDRESS");
        TEXT_PLACEHOLDER_TYPES.put("{IBAN}",          "PII_IBAN");
        TEXT_PLACEHOLDER_TYPES.put("{CREDIT_CARD}",   "PII_CREDIT_CARD");
    }

    // =====================================================================================
    // HARD NEGATIVES - v3 error analysis: 59% of the fine-tuned model's false positives
    // were USERNAME/PASSWORD flagged on code-like tokens in clean prose, plus PERSON on
    // placeholder organization names. These sentences contain exactly such tokens but are
    // labeled clean, teaching the model that reference codes, build ids, tracking numbers,
    // license keys, service accounts and placeholder orgs are NOT PII.
    // =====================================================================================
    private static final String[] HARD_NEGATIVE_TEXT_TEMPLATES = {
        "Your reference number is {CODE}; please quote it in all future correspondence.",
        "Use discount code {CODE} at checkout to get 15 percent off your next order.",
        "The failing build is {BUILD}; full logs are attached to the ticket.",
        "Your parcel is on the way, tracking code {TRACK}.",
        "Session {CODE} expired after 30 minutes of inactivity; simply sign in again.",
        "The temporary access code {CODE} is valid for the next 10 minutes.",
        "Invoice {REF} was generated automatically; no action is required.",
        "Deployment {BUILD} rolled out to the staging cluster without incident.",
        "Please install update {VERSION} before the end of the week.",
        "The license key {CODE} has been activated on two of the three allowed seats.",
        "Ticket {REF} has been merged into the master incident.",
        "Backup job {REF} completed with zero failed items; the service account svc_backup_01 rotated its credentials automatically.",
        "The scheduled task runs under the machine account svc_report_gen; no personal account is involved.",
        "{ORG} reserves the right to update these terms at any time.",
        "The proposal was submitted on behalf of {ORG} and is pending review.",
        "All inquiries should be directed to the {ORG} support desk.",
        "{ORG} employees must complete the annual compliance training by Friday.",
        "Voucher {CODE} can be redeemed once per account and expires at the end of the quarter."
    };

    private static final String[] PLACEHOLDER_ORGS = {
        "XYZ School", "ACME Corp", "ABC Company", "Example Inc.", "Contoso Ltd.", "the vendor"
    };

    /** Fills a hard-negative sentence with realistic machine-generated (non-PII) tokens. */
    private static String hardNegativeSentence() {
        String text = HARD_NEGATIVE_TEXT_TEMPLATES[random.nextInt(HARD_NEGATIVE_TEXT_TEMPLATES.length)];
        while (text.contains("{CODE}"))    { text = text.replaceFirst("\\{CODE\\}",    faker.bothify("??##-??##", true)); }
        while (text.contains("{REF}"))     { text = text.replaceFirst("\\{REF\\}",     "REQ-" + faker.numerify("######")); }
        while (text.contains("{BUILD}"))   { text = text.replaceFirst("\\{BUILD\\}",   faker.numerify("####") + "-rc" + faker.numerify("#")); }
        while (text.contains("{TRACK}"))   { text = text.replaceFirst("\\{TRACK\\}",   faker.bothify("1Z###??#########", true)); }
        while (text.contains("{VERSION}")) { text = text.replaceFirst("\\{VERSION\\}", "v" + faker.numerify("#.##.#")); }
        while (text.contains("{ORG}"))     { text = text.replaceFirst("\\{ORG\\}",     PLACEHOLDER_ORGS[random.nextInt(PLACEHOLDER_ORGS.length)]); }
        return text;
    }

    /** One neutral (non-PII) sentence: 50/50 mix of plain business prose and hard negatives. */
    private static String nonPiiSentence() {
        return random.nextBoolean()
                ? NON_PII_TEXT_TEMPLATES[random.nextInt(NON_PII_TEXT_TEMPLATES.length)]
                : hardNegativeSentence();
    }

    /** Business/system prose with no PII - teaches the model that prose is not automatically PII. */
    private static final String[] NON_PII_TEXT_TEMPLATES = {
        "The deployment completed successfully after three retries. No further action is required.",
        "Cache invalidation is scheduled for the next maintenance window. Expect brief latency spikes.",
        "Quarterly targets were exceeded in two regions. Detailed figures are in the attached report.",
        "The database migration finished without errors. All indexes were rebuilt overnight.",
        "Please review the updated onboarding checklist before the next sprint planning.",
        "Service health is back to normal. Root cause was a misconfigured load balancer rule.",
        "Reminder: the quarterly security training must be completed by the end of the month.",
        "Build 4812 passed all integration tests and is ready for the staging environment.",
        "The new pricing page went live this morning. Conversion metrics will be reviewed on Friday.",
        "Inventory sync ran twice due to a scheduler overlap; duplicates were removed automatically.",
        "API rate limits will increase from 100 to 250 requests per minute starting next release.",
        "The incident was resolved by rolling back to the previous configuration version.",
        "Documentation for the webhook retry policy has been updated and published.",
        "Storage usage reached 78 percent of quota. Consider archiving records older than two years.",
        "All scheduled jobs completed on time. Average processing latency was 340 milliseconds."
    };

    /** Fills one PII sentence template with faker values; records each value as an entity. */
    private static String piiSentence(String currentPath, ArrayNode entities) {
        String text = PII_TEXT_TEMPLATES[random.nextInt(PII_TEXT_TEMPLATES.length)];

        for (Map.Entry<String, String> ph : TEXT_PLACEHOLDER_TYPES.entrySet()) {
            while (text.contains(ph.getKey())) {
                String fakeValue = fakePiiValue(ph.getValue());
                text = text.replaceFirst(java.util.regex.Pattern.quote(ph.getKey()),
                        java.util.regex.Matcher.quoteReplacement(fakeValue));

                ObjectNode entityRecord = factory.objectNode();
                entityRecord.put("type", ph.getValue());
                entityRecord.put("value", fakeValue);
                entityRecord.put("json_path", currentPath);
                entities.add(entityRecord);
            }
        }
        return text;
    }

    /** Generates prose (1-2 sentences) with embedded PII for a JSON field value. */
    private static JsonNode generateFreeTextWithPii(String currentPath, ArrayNode entities) {
        String text = piiSentence(currentPath, entities);
        if (random.nextDouble() < 0.30) {
            text = text + " " + piiSentence(currentPath, entities);
        }
        return factory.textNode(text);
    }

    /**
     * Builds an ai4privacy-style PLAIN-TEXT record: the payload is bare prose
     * (2-5 sentences, like an email or support message), not a JSON object.
     * PII records mix PII-bearing and neutral sentences; non-PII records are
     * neutral prose only. Entities are recorded at json_path "$".
     */
    private static JsonNode plainTextPayload(boolean withPii, ArrayNode entities) {
        int sentenceCount = 2 + random.nextInt(4); // 2-5
        List<String> sentences = new ArrayList<>();

        if (withPii) {
            int piiSentences = 1 + random.nextInt(Math.max(1, sentenceCount - 1)); // at least 1
            for (int i = 0; i < sentenceCount; i++) {
                if (i < piiSentences) {
                    sentences.add(piiSentence("$", entities));
                } else {
                    //neutral filler can also be a hard negative - PII next to code-like
                    //tokens teaches the model to separate them within the same message
                    sentences.add(nonPiiSentence());
                }
            }
        } else {
            for (int i = 0; i < sentenceCount; i++) {
                sentences.add(nonPiiSentence());
            }
        }

        Collections.shuffle(sentences, random);
        return factory.textNode(String.join(" ", sentences));
    }

    /** Single faker value for the given PII type (shared by field-level and free-text generation). */
    private static String fakePiiValue(String piiType) {
        switch (piiType) {
            case "PII_PERSON_NAME": return faker.name().fullName();
            case "PII_PERSON_FIRST_NAME":
            case "PII_FIRST_NAME_FEMALE":
            case "PII_FIRST_NAME_MALE": return faker.name().firstName();
            case "PII_PERSON_LAST_NAME":
            case "PII_LAST_NAME_MALE":
            case "PII_LAST_NAME_FEMALE": return faker.name().lastName();
            case "PII_COMPANY_NAME":
            case "PII_ORGANIZATION_NAME": return faker.company().name();
            case "PII_USERNAME_ALIAS": return faker.superhero().name().replace(" ", "_");
            case "PII_EMAIL": return faker.internet().emailAddress();
            case "PII_PHONE_NUMBER": return faker.phoneNumber().cellPhone();
            case "PII_ADDRESS_LINE": return faker.address().streetAddress();
            case "PII_LOCATION_CITY": return faker.address().city();
            case "PII_LOCATION_COUNTRY": return faker.address().country();
            case "PII_LOCATION_STATE": return faker.address().state();
            case "PII_IBAN": return faker.finance().iban();
            case "PII_PASSWORD": return faker.internet().password();
            case "PII_NATIONAL_ID": return faker.idNumber().valid();
            case "PII_USERNAME": return faker.name().username();
            case "PII_IP_ADDRESS": return faker.internet().ipV4Address();
            case "PII_MAC_ADDRESS": return faker.internet().macAddress();
            case "PII_CREDIT_CARD": return faker.finance().creditCard();
            case "PII_DATE_OF_BIRTH":
                int year = 1950 + random.nextInt(50);
                return String.format("%04d-%02d-%02d", year, 1 + random.nextInt(12), 1 + random.nextInt(28));
            default: return "Sample_" + faker.lorem().word();
        }
    }

    private static JsonNode replaceWithFakeData(String val, String currentPath, ArrayNode entities) {
        if (val.equals("PII_FREE_TEXT")) {
            return generateFreeTextWithPii(currentPath, entities);
        }

        if (val.startsWith("PII_")) {
            String fakeValue = fakePiiValue(val);

            // Register the generated PII entity
            ObjectNode entityRecord = factory.objectNode();
            entityRecord.put("type", val);
            entityRecord.put("value", fakeValue);
            entityRecord.put("json_path", currentPath);
            entities.add(entityRecord);

            return factory.textNode(fakeValue);
        }

        // Process standard system types (No entities recorded)
        switch (val) {
            case "STRING": return factory.textNode(realisticString(currentPath));
            case "TEXT": return factory.textNode(NON_PII_TEXT_TEMPLATES[random.nextInt(NON_PII_TEXT_TEMPLATES.length)]);
            case "INT": return factory.numberNode(faker.number().numberBetween(1, 99999));
            case "DECIMAL": return factory.numberNode(Math.round(faker.number().randomDouble(2, 1, 5000) * 100.0) / 100.0);
            case "BOOLEAN": return factory.booleanNode(random.nextBoolean());
            case "DATE": {
                int dateYear = 2015 + random.nextInt(10);
                return factory.textNode(String.format("%04d-%02d-%02d", dateYear, 1 + random.nextInt(12), 1 + random.nextInt(28)));
            }

            // Generate a random timestamp within the last 10 years
            case "DATETIME": {
                long tenYearsInMillis = 10L * 365 * 24 * 60 * 60 * 1000;
                long randomEpoch = System.currentTimeMillis() - (long) (random.nextDouble() * tenYearsInMillis);
                return factory.textNode(ISO_FORMATTER.format(Instant.ofEpochMilli(randomEpoch)));
            }
            case "DATETIME_EPOCH": {
                long tenYearsInMillis = 10L * 365 * 24 * 60 * 60 * 1000;
                long randomEpoch = System.currentTimeMillis() - (long) (random.nextDouble() * tenYearsInMillis);
                return factory.numberNode(randomEpoch / 1000);
            }
            case "URL": return factory.textNode(faker.internet().url());
            case "ARRAY": return factory.arrayNode();
            case "OBJECT":
                ObjectNode obj = factory.objectNode();
                obj.put("id", faker.number().numberBetween(100, 900));
                return obj;
            case "NULL": return NullNode.getInstance();
        }

        return factory.textNode(val);
    }
}