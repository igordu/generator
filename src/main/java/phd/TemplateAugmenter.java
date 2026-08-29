package phd;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.*;

import java.io.File;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * The {@code TemplateAugmenter} class is a data augmentation and fuzzing tool designed to
 * automatically generate a large dataset of unique, mutated variations from base JSON templates.
 *
 * <p><b>Main Execution Pipeline:</b></p>
 * <ol>
 * <li><b>Initialization:</b> Clears the output directory ({@code generated_templates}) and loads base JSON files from the {@code templates} directory.</li>
 * <li><b>Generation Loop:</b> For each input template, the algorithm attempts to generate a target number of unique variations (default is 100).</li>
 * <li><b>Uniqueness Guarantee:</b> Every mutated JSON is serialized into a string and checked against a global {@code HashSet} ({@code uniqueStructures}). Duplicates are immediately discarded.</li>
 * <li><b>Categorization & Saving:</b> The algorithm scans the final mutated JSON for PII (Personally Identifiable Information). Files are then saved sequentially with either a {@code PII_} or {@code NON_PII_} prefix.</li>
 * </ol>
 *
 * <p><b>Data Mutation Strategies ({@code mutateNode}):</b></p>
 * The algorithm recursively traverses the JSON tree and applies several probabilistic mutations:
 * <ul>
 * <li><b>Dropping (10% chance):</b> Removes fields from objects or elements from arrays, simulating missing data. <i>(Protected if the node contains PII)</i>.</li>
 * <li><b>Key Fuzzing (30% chance):</b> Renames object keys to common aliases (e.g., mapping 'email' to 'emailAddress') or injects noise (e.g., adding a 'v_' prefix or '_data' suffix).</li>
 * <li><b>Noise Injection (50% chance):</b> Injects new, safe dummy fields with random values into objects (e.g., {@code {"item_count": 42}} or {@code {"is_active": true}}).</li>
 * <li><b>Shuffling:</b> Randomizes the order of keys within objects and the order of elements within arrays.</li>
 * <li><b>Value Corruption:</b> Standard text values have a small probability of being replaced by {@code null} (2%), the literal string {@code "NULL"} (3%), or an empty string {@code ""} (2%).</li>
 * </ul>
 *
 * <p><b>Structural Augmentations (Applied to the Root Payload):</b></p>
 * <ul>
 * <li><b>Deep Nesting (5% chance):</b> Wraps the entire JSON payload inside three layers of random log/telemetry keys (e.g., {@code server_logs -> request_context -> raw_data}).</li>
 * <li><b>API Envelopes (20% chance):</b> Wraps the data inside standard simulated API response structures, such as a basic REST success payload, a paginated response with metadata, or a GraphQL-style {@code data} wrapper.</li>
 * </ul>
 *
 * <p><b>PII Handling & Protection:</b></p>
 * The algorithm uses the {@code containsPii} method to recursively search for text values starting with the prefix {@code "PII_"}.
 * Any node containing PII is strictly protected against deletion or value corruption to ensure the integrity of the target data being tested.
 */
public class TemplateAugmenter {

    // Number of unique variations to generate for EACH original template
    private static final int VARIATIONS_PER_TEMPLATE = 20;

    private static final ObjectMapper mapper = new ObjectMapper();
    private static final JsonNodeFactory factory = JsonNodeFactory.instance;
    private static final Random random = new Random();

    // Global counters for sequential file naming
    private static int piiCounter = 1;
    private static int nonPiiCounter = 1;

    public static void main(String[] args) throws Exception {
        Path inputDir = Paths.get(Objects.requireNonNull( TemplateAugmenter.class.getClassLoader().getResource("templates") ).toURI());
        Path outputDir = Paths.get("generated_templates");

        // Ensure directories exist
        if (!Files.exists(inputDir)) Files.createDirectories(inputDir);
        if (!Files.exists(outputDir)) Files.createDirectories(outputDir);

        // Delete old templates
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(outputDir)) {
            for (Path entry : stream) {
                Files.deleteIfExists(entry);
            }
        }

        File[] templates = inputDir.toFile().listFiles();
        if (templates == null || templates.length == 0) {
            System.out.println("No JSON files found in the 'templates' directory.");
            return;
        }

        // Set to keep track of generated JSON strings to ensure 100% uniqueness
        Set<String> uniqueStructures = new HashSet<>();

        for (File templateFile : templates) {
            JsonNode rootNode = mapper.readTree(templateFile);

            int variationsGenerated = 0;
            int maxAttempts = VARIATIONS_PER_TEMPLATE * 10;
            int attempts = 0;

            System.out.println("Processing template: " + templateFile.getName());

            while (variationsGenerated < VARIATIONS_PER_TEMPLATE && attempts < maxAttempts) {
                attempts++;

                JsonNode mutatedNode = mutateNode(rootNode.deepCopy());

                // Deep Nesting
                if (random.nextDouble() < 0.05) {
                    mutatedNode = applyDeepNesting(mutatedNode);
                }

                // API Envelopes
                if (random.nextDouble() < 0.20) {
                    mutatedNode = wrapInEnvelope(mutatedNode);
                }

                String jsonStringRepresentation = mapper.writeValueAsString(mutatedNode);

                if (!uniqueStructures.contains(jsonStringRepresentation)) {
                    uniqueStructures.add(jsonStringRepresentation);

                    boolean isPii = containsPii(mutatedNode);

                    String fileName = isPii
                            ? String.format("PII_%05d.json", piiCounter++)
                            : String.format("NON_PII_%05d.json", nonPiiCounter++);

                    File outputFile = new File(outputDir.toFile(), fileName);
                    mapper.writerWithDefaultPrettyPrinter().writeValue(outputFile, mutatedNode);

                    variationsGenerated++;
                }
            }

            if (variationsGenerated < VARIATIONS_PER_TEMPLATE) {
                System.out.println("Warning: Could only generate " + variationsGenerated +
                        " unique variations for " + templateFile.getName());
            }
        }
        System.out.println("Generation complete! Total unique files created: " + uniqueStructures.size());
        System.out.println("-> PII files: " + (piiCounter - 1));
        System.out.println("-> NON_PII files: " + (nonPiiCounter - 1));
    }

    private static JsonNode mutateNode(JsonNode node) {
        if (node.isObject()) {
            ObjectNode objNode = (ObjectNode) node;
            List<Map.Entry<String, JsonNode>> fieldsList = new ArrayList<>();
            Iterator<Map.Entry<String, JsonNode>> iter = objNode.fields();

            while (iter.hasNext()) {
                Map.Entry<String, JsonNode> entry = iter.next();
                String key = entry.getKey();
                JsonNode value = entry.getValue();

                if (!containsPii(value) && random.nextDouble() < 0.10) {
                    continue;
                }

                if (random.nextDouble() < 0.30) {
                    key = fuzzKey(key);
                }

                fieldsList.add(new AbstractMap.SimpleEntry<>(key, mutateNode(value)));
            }

            if (random.nextDouble() < 0.50) {
                fieldsList.add(generateSafeNoiseEntry());
            }

            Collections.shuffle(fieldsList);

            ObjectNode newObjNode = factory.objectNode();
            for (Map.Entry<String, JsonNode> entry : fieldsList) {
                newObjNode.set(entry.getKey(), entry.getValue());
            }
            return newObjNode;

        } else if (node.isArray()) {
            ArrayNode arrNode = (ArrayNode) node;
            ArrayNode newArrNode = factory.arrayNode();

            for (JsonNode item : arrNode) {

                if (arrNode.size() > 1 && !containsPii(item) && random.nextDouble() < 0.10) {
                    continue;
                }
                newArrNode.add(mutateNode(item));
            }

            List<JsonNode> arrList = new ArrayList<>();
            newArrNode.elements().forEachRemaining(arrList::add);
            Collections.shuffle(arrList);
            newArrNode.removeAll();
            newArrNode.addAll(arrList);
            return newArrNode;

        } else if (node.isTextual()) {
            if (node.asText().startsWith("PII_")) {
                return node;
            }
            double chance = random.nextDouble();
            if (chance < 0.02) {
                return NullNode.getInstance();
            } else if (chance < 0.05) {
                return factory.textNode("NULL");
            } else if (chance < 0.07) {
                return factory.textNode("");
            }
            return node;
        }

        return node;
    }

    private static JsonNode applyDeepNesting(JsonNode payload) {
        String[] l1Keys = {"server_logs", "telemetry_data", "system_state", "audit_trail"};
        String[] l2Keys = {"transaction_trace", "request_context", "event_data", "payload_wrapper"};
        String[] l3Keys = {"debug_payload", "raw_data", "extracted_info", "body"};

        ObjectNode root = factory.objectNode();
        ObjectNode level1 = factory.objectNode();
        ObjectNode level2 = factory.objectNode();

        level2.set(l3Keys[random.nextInt(l3Keys.length)], payload);
        level1.set(l2Keys[random.nextInt(l2Keys.length)], level2);
        root.set(l1Keys[random.nextInt(l1Keys.length)], level1);

        return root;
    }

    private static JsonNode wrapInEnvelope(JsonNode originalData) {
        ObjectNode envelope = factory.objectNode();
        int envelopeType = random.nextInt(3);

        switch (envelopeType) {
            case 0:
                envelope.put("status", "success");
                envelope.put("code", 200);
                envelope.set("data", originalData);
                break;
            case 1:
                ObjectNode meta = factory.objectNode();
                meta.put("page", 1);
                meta.put("total_pages", 5);
                envelope.set("metadata", meta);
                envelope.set("results", originalData);
                break;
            case 2:
                ObjectNode dataNode = factory.objectNode();
                dataNode.set("fetchResource", originalData);
                envelope.set("data", dataNode);
                break;
        }
        return envelope;
    }

    private static Map.Entry<String, JsonNode> generateSafeNoiseEntry() {
        String[] safeIntKeys = {"item_count", "retry_attempts", "system_ping_ms", "http_status", "priority_level", "stock_quantity", "page_index"};
        String[] safeBoolKeys = {"is_active", "has_discount", "auto_renew", "is_deleted", "cache_hit", "requires_update"};

        // Each double key has its own realistic range [min, max]
        String[][] safeDoubleEntries = {
            {"price",             "0",    "9999"},
            {"discount_rate",     "0",    "100"},
            {"temperature_c",     "-20",  "50"},
            {"cpu_usage_percent", "0",    "100"},
            {"conversion_rate",   "0",    "10"},
            {"response_time_sec", "0",    "30"}
        };

        int noiseType = random.nextInt(3);
        String key;
        JsonNode value;

        if (noiseType == 0) {
            key = safeIntKeys[random.nextInt(safeIntKeys.length)];
            value = factory.numberNode(random.nextInt(5000));
        } else if (noiseType == 1) {
            String[] entry = safeDoubleEntries[random.nextInt(safeDoubleEntries.length)];
            key = entry[0];
            double min = Double.parseDouble(entry[1]);
            double max = Double.parseDouble(entry[2]);
            double randomDouble = min + random.nextDouble() * (max - min);
            value = factory.numberNode(Math.round(randomDouble * 100.0) / 100.0);
        } else {
            key = safeBoolKeys[random.nextInt(safeBoolKeys.length)];
            value = factory.booleanNode(random.nextBoolean());
        }

        return new AbstractMap.SimpleEntry<>(key, value);
    }

    private static String fuzzKey(String originalKey) {
        String lowerKey = originalKey.toLowerCase();

        if (lowerKey.contains("name")) {
            String[] options = {"fullName", "clientName", "userName", "name_val"};
            return options[random.nextInt(options.length)];
        }
        if (lowerKey.contains("email")) {
            String[] options = {"emailAddress", "contactEmail", "mail", "primary_email"};
            return options[random.nextInt(options.length)];
        }
        if (lowerKey.contains("id")) {
            String[] options = {"uuid", "identifier", "recordId", "primaryKey"};
            return options[random.nextInt(options.length)];
        }
        if (lowerKey.contains("phone")) {
            String[] options = {"mobile", "contactNumber", "telephone", "cell"};
            return options[random.nextInt(options.length)];
        }

        if (random.nextBoolean()) {
            return "v_" + originalKey;
        } else {
            return originalKey + "_data";
        }
    }

    private static boolean containsPii(JsonNode node) {
        if (node.isObject()) {
            Iterator<JsonNode> iter = node.elements();
            while (iter.hasNext()) {
                if (containsPii(iter.next())) return true;
            }
        } else if (node.isArray()) {
            for (JsonNode item : node) {
                if (containsPii(item)) return true;
            }
        } else if (node.isTextual()) {
            String val = node.asText();
            if (val != null && val.startsWith("PII_")) {
                return true;
            }
        }
        return false;
    }
}