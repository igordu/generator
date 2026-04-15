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

    private static final int TOTAL_RECORDS = 5000;
    private static final double PII_RATIO = 0.70;
    private static final ObjectMapper mapper = new ObjectMapper();
    private static final Faker faker = new Faker(new Locale("en-US"));
    private static final Random random = new Random();
    private static final JsonNodeFactory factory = JsonNodeFactory.instance;
    private static final String OUTPUT_FILE = "generated_data/dataset_with_labels.jsonl";
    private static final String OUTPUT_REPORT_FILE = "generated_data/dataset_report.txt";
    private static final String INPUT_DIR = "generated_templates";
    private static final List SYSTEM_TYPES = Arrays.asList("STRING", "INT", "DECIMAL", "BOOLEAN", "DATE", "DATETIME", "DATETIME_EPOCH", "URL", "OBJECT");

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

        System.out.println("Starting dataset generation. Target PII: " + targetPiiCount + ", Target Non-PII: " + targetNonPiiCount);

        // REJECTION SAMPLING LOOP: Guarantees ground-truth labels and strict dataset ratios
        while (generatedPii < targetPiiCount || generatedNonPii < targetNonPiiCount) {
            // Pick a random template
            JsonNode template = allTemplates.get(random.nextInt(allTemplates.size()));
            ArrayNode entities = factory.arrayNode();

            // Generate without any conditioning or masking to ensure honest output
            JsonNode filledPayload = fillTemplate(template.deepCopy(), "$", entities);
            boolean isActuallyPii = entities.size() > 0;

            // Bucket the record only if the target quota for its type is not yet full
            if (isActuallyPii && generatedPii < targetPiiCount) {
                finalDataset.add(wrapRecord(filledPayload, entities, true));
                generatedPii++;

                // Track entities that successfully entered the dataset
                for (JsonNode entity : entities) {
                    String type = entity.get("type").asText();
                    piiDistribution.put(type, piiDistribution.getOrDefault(type, 0) + 1);
                }

            } else if (!isActuallyPii && generatedNonPii < targetNonPiiCount) {
                finalDataset.add(wrapRecord(filledPayload, entities, false));
                generatedNonPii++;
            }
        }

        Collections.shuffle(finalDataset);

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(OUTPUT_FILE))) {
            for (String record : finalDataset) {
                writer.write(record);
                writer.newLine();
            }
        }

        System.out.println("Dataset generation complete! Strict ratio enforced: " + OUTPUT_FILE);

        // Print the detailed analytical report
        printReport(generatedPii, generatedNonPii, piiDistribution);
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
                int randomSize = 1 + random.nextInt(4); // Generates 1 to 4 elements
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

    private static JsonNode replaceWithFakeData(String val, String currentPath, ArrayNode entities) {
        if (val.startsWith("PII_")) {
            String fakeValue;
            switch (val) {
                case "PII_PERSON_NAME": fakeValue = faker.name().fullName(); break;
                case "PII_PERSON_FIRST_NAME":
                case "PII_FIRST_NAME_FEMALE":
                case "PII_FIRST_NAME_MALE": fakeValue = faker.name().firstName(); break;
                case "PII_PERSON_LAST_NAME":
                case "PII_LAST_NAME_MALE":
                case "PII_LAST_NAME_FEMALE": fakeValue = faker.name().lastName(); break;
                case "PII_COMPANY_NAME":
                case "PII_ORGANIZATION_NAME": fakeValue = faker.company().name(); break;
                case "PII_USERNAME_ALIAS": fakeValue = faker.superhero().name().replace(" ", "_"); break;
                case "PII_EMAIL": fakeValue = faker.internet().emailAddress(); break;
                case "PII_PHONE_NUMBER": fakeValue = faker.phoneNumber().cellPhone(); break;
                case "PII_ADDRESS_LINE": fakeValue = faker.address().streetAddress(); break;
                case "PII_LOCATION_CITY": fakeValue = faker.address().city(); break;
                case "PII_LOCATION_COUNTRY": fakeValue = faker.address().country(); break;
                case "PII_IBAN": fakeValue = faker.finance().iban(); break;
                case "PII_PASSWORD": fakeValue = faker.internet().password(); break;
                case "PII_NATIONAL_ID": fakeValue = faker.idNumber().valid(); break;
                case "PII_USERNAME": fakeValue = faker.name().username(); break;
                case "PII_DATE_OF_BIRTH":
                    int year = 1950 + random.nextInt(50);
                    fakeValue = String.format("%04d-%02d-%02d", year, 1 + random.nextInt(12), 1 + random.nextInt(28));
                    break;
                default: fakeValue = "Sample_" + faker.lorem().word();
            }

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
            case "STRING": return factory.textNode(faker.lorem().word());
            case "INT": return factory.numberNode(faker.number().numberBetween(1, 99999));
            case "DECIMAL": return factory.numberNode(Math.round(faker.number().randomDouble(2, 1, 5000) * 100.0) / 100.0);
            case "BOOLEAN": return factory.booleanNode(random.nextBoolean());
            case "DATE": return factory.textNode(String.format("202%d-%02d-%02d", random.nextInt(5), 1 + random.nextInt(12), 1 + random.nextInt(28)));

            // Generate a random timestamp within the last 10 years
            case "DATETIME":
            case "DATETIME_EPOCH": {
                long tenYearsInMillis = 10L * 365 * 24 * 60 * 60 * 1000;
                long randomEpoch = System.currentTimeMillis() - (long) (random.nextDouble() * tenYearsInMillis);
                return factory.textNode(ISO_FORMATTER.format(Instant.ofEpochMilli(randomEpoch)));
            }
            case "URL": return factory.textNode(faker.internet().url());
            case "OBJECT":
                ObjectNode obj = factory.objectNode();
                obj.put("id", faker.number().numberBetween(100, 900));
                return obj;
            case "NULL": return NullNode.getInstance();
        }

        return factory.textNode(val);
    }
}