# # # from pathlib import Path
# # # import sys

# # # import torch
# # # import torch.nn as nn

# # # from torchvision import transforms
# # # from torchvision.models import mobilenet_v3_small
# # # from torchvision import models

# # # from PIL import Image
# # # from ultralytics import YOLO


# # # # ============================================================
# # # # PATHS
# # # # ============================================================

# # # BASE_DIR = Path(__file__).resolve().parents[1]


# # # # ------------------------------------------------------------
# # # # WASTE GATE V2
# # # # ------------------------------------------------------------

# # # WASTE_GATE_MODEL = (
# # #     BASE_DIR
# # #     / "models"
# # #     / "waste_gate_v2_best.pth"
# # # )


# # # # ------------------------------------------------------------
# # # # YOLO WASTE TYPE MODEL
# # # # ------------------------------------------------------------

# # # YOLO_MODEL = (
# # #     BASE_DIR
# # #     / "models"
# # #     / "waste_types_training"
# # #     / "yolo11n_waste_types"
# # #     / "weights"
# # #     / "best.pt"
# # # )


# # # # ------------------------------------------------------------
# # # # SCENE ANALYSIS MODEL
# # # # ------------------------------------------------------------

# # # SCENE_MODEL = (
# # #     BASE_DIR
# # #     / "models"
# # #     / "scene_analysis_resnet18_best.pth"
# # # )


# # # # ============================================================
# # # # SETTINGS
# # # # ============================================================

# # # # V2 performed best at 0.40 according to your threshold analysis
# # # WASTE_THRESHOLD = 0.40

# # # YOLO_CONFIDENCE = 0.50

# # # SCENE_CONFIDENCE = 0.50


# # # DEVICE = torch.device(
# # #     "cuda"
# # #     if torch.cuda.is_available()
# # #     else "cpu"
# # # )


# # # # ============================================================
# # # # SCENE CLASSES
# # # # ============================================================

# # # SCENE_CLASSES = [
# # #     "construction_debris",
# # #     "drain_blockage",
# # #     "garbage_dump",
# # #     "overflowing_bin"
# # # ]


# # # # ============================================================
# # # # SCENE TRANSFORM
# # # # ============================================================

# # # scene_transform = transforms.Compose([

# # #     transforms.Resize(
# # #         (224, 224)
# # #     ),

# # #     transforms.ToTensor(),

# # #     transforms.Normalize(
# # #         mean=[
# # #             0.485,
# # #             0.456,
# # #             0.406
# # #         ],
# # #         std=[
# # #             0.229,
# # #             0.224,
# # #             0.225
# # #         ]
# # #     )
# # # ])


# # # # ============================================================
# # # # WASTE GATE TRANSFORM
# # # # ============================================================

# # # waste_gate_transform = transforms.Compose([

# # #     transforms.Resize(
# # #         (224, 224)
# # #     ),

# # #     transforms.ToTensor(),

# # #     transforms.Normalize(
# # #         mean=[
# # #             0.485,
# # #             0.456,
# # #             0.406
# # #         ],
# # #         std=[
# # #             0.229,
# # #             0.224,
# # #             0.225
# # #         ]
# # #     )
# # # ])


# # # # ============================================================
# # # # LOAD WASTE GATE V2
# # # # ============================================================

# # # def load_waste_gate():

# # #     print(
# # #         "Loading Waste Gate V2..."
# # #     )

# # #     model = mobilenet_v3_small(
# # #         weights=None,
# # #         num_classes=2
# # #     )

# # #     checkpoint = torch.load(
# # #         WASTE_GATE_MODEL,
# # #         map_location=DEVICE
# # #     )

# # #     if (
# # #         isinstance(checkpoint, dict)
# # #         and "model_state_dict" in checkpoint
# # #     ):

# # #         state_dict = (
# # #             checkpoint["model_state_dict"]
# # #         )

# # #     elif (
# # #         isinstance(checkpoint, dict)
# # #         and "state_dict" in checkpoint
# # #     ):

# # #         state_dict = (
# # #             checkpoint["state_dict"]
# # #         )

# # #     else:

# # #         state_dict = checkpoint

# # #     model.load_state_dict(
# # #         state_dict
# # #     )

# # #     model.to(DEVICE)

# # #     model.eval()

# # #     return model


# # # # ============================================================
# # # # LOAD YOLO
# # # # ============================================================

# # # def load_yolo():

# # #     print(
# # #         "Loading multi-class YOLO detector..."
# # #     )

# # #     return YOLO(
# # #         str(YOLO_MODEL)
# # #     )


# # # # ============================================================
# # # # LOAD SCENE MODEL
# # # # ============================================================

# # # def load_scene_model():

# # #     print(
# # #         "Loading scene analysis model..."
# # #     )

# # #     checkpoint = torch.load(
# # #         SCENE_MODEL,
# # #         map_location=DEVICE
# # #     )

# # #     classes = checkpoint.get(
# # #         "classes",
# # #         SCENE_CLASSES
# # #     )

# # #     # --------------------------------------------------------
# # #     # ResNet18 backbone
# # #     # --------------------------------------------------------

# # #     resnet = models.resnet18(
# # #         weights=None
# # #     )

# # #     # Remove final classifier
# # #     feature_extractor = nn.Sequential(
# # #         *list(resnet.children())[:-1]
# # #     )

# # #     feature_extractor = (
# # #         feature_extractor.to(DEVICE)
# # #     )

# # #     feature_extractor.eval()

# # #     # --------------------------------------------------------
# # #     # Classifier
# # #     # --------------------------------------------------------

# # #     feature_size = (
# # #         resnet.fc.in_features
# # #     )

# # #     classifier = nn.Linear(
# # #         feature_size,
# # #         len(classes)
# # #     )

# # #     classifier.load_state_dict(
# # #         checkpoint[
# # #             "classifier_state_dict"
# # #         ]
# # #     )

# # #     classifier = (
# # #         classifier.to(DEVICE)
# # #     )

# # #     classifier.eval()

# # #     return (
# # #         feature_extractor,
# # #         classifier,
# # #         classes
# # #     )


# # # # ============================================================
# # # # WASTE GATE PREDICTION
# # # # ============================================================

# # # def check_waste(
# # #     model,
# # #     image
# # # ):

# # #     image_tensor = (
# # #         waste_gate_transform(image)
# # #         .unsqueeze(0)
# # #         .to(DEVICE)
# # #     )

# # #     with torch.no_grad():

# # #         output = model(
# # #             image_tensor
# # #         )

# # #         probabilities = torch.softmax(
# # #             output,
# # #             dim=1
# # #         )[0]

# # #     # 0 = non_waste
# # #     # 1 = waste

# # #     non_waste_probability = (
# # #         probabilities[0].item()
# # #     )

# # #     waste_probability = (
# # #         probabilities[1].item()
# # #     )

# # #     is_waste = (
# # #         waste_probability
# # #         >= WASTE_THRESHOLD
# # #     )

# # #     return (
# # #         is_waste,
# # #         waste_probability,
# # #         non_waste_probability
# # #     )


# # # # ============================================================
# # # # YOLO ANALYSIS
# # # # ============================================================

# # # def analyze_waste(
# # #     yolo_model,
# # #     image_path
# # # ):

# # #     results = yolo_model.predict(
# # #         source=str(image_path),
# # #         conf=YOLO_CONFIDENCE,
# # #         verbose=False
# # #     )

# # #     result = results[0]

# # #     detections = []

# # #     if result.boxes is None:

# # #         return (
# # #             detections,
# # #             result.orig_shape
# # #         )

# # #     names = result.names

# # #     for box in result.boxes:

# # #         class_id = int(
# # #             box.cls[0].item()
# # #         )

# # #         confidence = float(
# # #             box.conf[0].item()
# # #         )

# # #         xyxy = (
# # #             box.xyxy[0]
# # #             .tolist()
# # #         )

# # #         detections.append({

# # #             "type": names[class_id],

# # #             "confidence":
# # #                 confidence,

# # #             "box":
# # #                 xyxy
# # #         })

# # #     return (
# # #         detections,
# # #         result.orig_shape
# # #     )


# # # # ============================================================
# # # # SCENE ANALYSIS
# # # # ============================================================

# # # def analyze_scene(
# # #     feature_extractor,
# # #     classifier,
# # #     classes,
# # #     image
# # # ):

# # #     image_tensor = (
# # #         scene_transform(image)
# # #         .unsqueeze(0)
# # #         .to(DEVICE)
# # #     )

# # #     with torch.no_grad():

# # #         features = feature_extractor(
# # #             image_tensor
# # #         )

# # #         features = features.view(
# # #             features.size(0),
# # #             -1
# # #         )

# # #         outputs = classifier(
# # #             features
# # #         )

# # #         probabilities = torch.softmax(
# # #             outputs,
# # #             dim=1
# # #         )[0]

# # #         predicted_index = int(
# # #             torch.argmax(
# # #                 probabilities
# # #             ).item()
# # #         )

# # #     scene = classes[
# # #         predicted_index
# # #     ]

# # #     confidence = probabilities[
# # #         predicted_index
# # #     ].item()

# # #     return (
# # #         scene,
# # #         confidence,
# # #         probabilities
# # #     )


# # # # ============================================================
# # # # SCENE-BASED AMOUNT ESTIMATION
# # # # ============================================================

# # # def estimate_scene_amount(
# # #     scene
# # # ):

# # #     # --------------------------------------------------------
# # #     # These are scene-based estimates.
# # #     #
# # #     # They represent visible accumulation level, not exact
# # #     # physical weight or volume.
# # #     # --------------------------------------------------------

# # #     scene_amounts = {

# # #         "garbage_dump":
# # #             "VERY LARGE",

# # #         "overflowing_bin":
# # #             "LARGE",

# # #         "construction_debris":
# # #             "LARGE",

# # #         "drain_blockage":
# # #             "MEDIUM"
# # #     }

# # #     return scene_amounts.get(
# # #         scene,
# # #         "MEDIUM"
# # #     )


# # # # ============================================================
# # # # VEHICLE RECOMMENDATION
# # # # ============================================================

# # # def recommend_vehicle(
# # #     amount
# # # ):

# # #     recommendations = {

# # #         "SMALL": {

# # #             "vehicle":
# # #                 "Small waste collection vehicle",

# # #             "reason":
# # #                 "Low visible waste accumulation."
# # #         },

# # #         "MEDIUM": {

# # #             "vehicle":
# # #                 "Medium waste collection vehicle",

# # #             "reason":
# # #                 "Moderate visible waste accumulation."
# # #         },

# # #         "LARGE": {

# # #             "vehicle":
# # #                 "Large waste collection vehicle",

# # #             "reason":
# # #                 "Large visible waste accumulation."
# # #         },

# # #         "VERY LARGE": {

# # #             "vehicle":
# # #                 "Large-capacity waste truck",

# # #             "reason":
# # #                 "Very large waste accumulation requires a high-capacity vehicle."
# # #         }
# # #     }

# # #     return recommendations.get(
# # #         amount,
# # #         {
# # #             "vehicle":
# # #                 "Vehicle recommendation unavailable",

# # #             "reason":
# # #                 "Unknown waste amount category."
# # #         }
# # #     )


# # # # ============================================================
# # # # UNION AREA CALCULATION
# # # # ============================================================

# # # def calculate_union_area(
# # #     boxes,
# # #     image_width,
# # #     image_height
# # # ):

# # #     if not boxes:

# # #         return 0.0

# # #     rectangles = []

# # #     for box in boxes:

# # #         x1, y1, x2, y2 = box

# # #         x1 = max(
# # #             0,
# # #             min(x1, image_width)
# # #         )

# # #         y1 = max(
# # #             0,
# # #             min(y1, image_height)
# # #         )

# # #         x2 = max(
# # #             0,
# # #             min(x2, image_width)
# # #         )

# # #         y2 = max(
# # #             0,
# # #             min(y2, image_height)
# # #         )

# # #         if (
# # #             x2 > x1
# # #             and y2 > y1
# # #         ):

# # #             rectangles.append(
# # #                 (x1, y1, x2, y2)
# # #             )

# # #     if not rectangles:

# # #         return 0.0

# # #     x_values = set()

# # #     for (
# # #         x1,
# # #         y1,
# # #         x2,
# # #         y2
# # #     ) in rectangles:

# # #         x_values.add(x1)
# # #         x_values.add(x2)

# # #     x_values = sorted(
# # #         x_values
# # #     )

# # #     total_area = 0.0

# # #     for i in range(
# # #         len(x_values) - 1
# # #     ):

# # #         left = x_values[i]

# # #         right = x_values[
# # #             i + 1
# # #         ]

# # #         width = (
# # #             right - left
# # #         )

# # #         if width <= 0:

# # #             continue

# # #         active_intervals = []

# # #         for (
# # #             x1,
# # #             y1,
# # #             x2,
# # #             y2
# # #         ) in rectangles:

# # #             if (
# # #                 x1 < right
# # #                 and x2 > left
# # #             ):

# # #                 active_intervals.append(
# # #                     (y1, y2)
# # #                 )

# # #         if not active_intervals:

# # #             continue

# # #         active_intervals.sort()

# # #         covered_height = 0.0

# # #         current_start = (
# # #             active_intervals[0][0]
# # #         )

# # #         current_end = (
# # #             active_intervals[0][1]
# # #         )

# # #         for (
# # #             start,
# # #             end
# # #         ) in active_intervals[1:]:

# # #             if start <= current_end:

# # #                 current_end = max(
# # #                     current_end,
# # #                     end
# # #                 )

# # #             else:

# # #                 covered_height += (
# # #                     current_end
# # #                     - current_start
# # #                 )

# # #                 current_start = start

# # #                 current_end = end

# # #         covered_height += (
# # #             current_end
# # #             - current_start
# # #         )

# # #         total_area += (
# # #             width
# # #             * covered_height
# # #         )

# # #     return total_area


# # # # ============================================================
# # # # VISIBLE WASTE COVERAGE
# # # # ============================================================

# # # def calculate_waste_coverage(
# # #     detections,
# # #     image_width,
# # #     image_height
# # # ):

# # #     boxes = [
# # #         detection["box"]
# # #         for detection in detections
# # #     ]

# # #     waste_area = (
# # #         calculate_union_area(
# # #             boxes,
# # #             image_width,
# # #             image_height
# # #         )
# # #     )

# # #     total_image_area = (
# # #         image_width
# # #         * image_height
# # #     )

# # #     if total_image_area <= 0:

# # #         return 0.0

# # #     coverage = (
# # #         waste_area
# # #         / total_image_area
# # #     ) * 100

# # #     return coverage


# # # # ============================================================
# # # # YOLO-BASED AMOUNT ESTIMATION
# # # # ============================================================

# # # def estimate_waste_amount(
# # #     coverage,
# # #     detections
# # # ):

# # #     object_count = len(
# # #         detections
# # #     )

# # #     if object_count == 1:

# # #         return "SMALL"

# # #     elif object_count <= 3:

# # #         if coverage < 25:

# # #             return "SMALL"

# # #         else:

# # #             return "MEDIUM"

# # #     elif object_count <= 7:

# # #         if coverage < 20:

# # #             return "MEDIUM"

# # #         else:

# # #             return "LARGE"

# # #     elif object_count <= 15:

# # #         if coverage < 20:

# # #             return "LARGE"

# # #         else:

# # #             return "VERY LARGE"

# # #     else:

# # #         return "VERY LARGE"


# # # # ============================================================
# # # # MAIN ANALYSIS
# # # # ============================================================

# # # def analyze_image(
# # #     image_path
# # # ):

# # #     image_path = Path(
# # #         image_path
# # #     )

# # #     if not image_path.exists():

# # #         print(
# # #             "ERROR: Image not found:"
# # #         )

# # #         print(
# # #             image_path
# # #         )

# # #         return

# # #     print()
# # #     print("=" * 60)
# # #     print(
# # #         "SWACHHLENS IMAGE ANALYSIS"
# # #     )
# # #     print("=" * 60)

# # #     print()

# # #     print(
# # #         f"Image: "
# # #         f"{image_path.name}"
# # #     )

# # #     print(
# # #         f"Device: "
# # #         f"{DEVICE}"
# # #     )


# # #     # ========================================================
# # #     # OPEN IMAGE
# # #     # ========================================================

# # #     try:

# # #         image = Image.open(
# # #             image_path
# # #         ).convert("RGB")

# # #     except Exception as e:

# # #         print()

# # #         print(
# # #             f"ERROR: Could not open image: {e}"
# # #         )

# # #         return


# # #     # ========================================================
# # #     # LOAD MODELS
# # #     # ========================================================

# # #     waste_gate = (
# # #         load_waste_gate()
# # #     )

# # #     yolo_model = (
# # #         load_yolo()
# # #     )

# # #     scene_feature_extractor, scene_classifier, scene_classes = (
# # #         load_scene_model()
# # #     )


# # #     # ========================================================
# # #     # STEP 1: WASTE GATE
# # #     # ========================================================

# # #     print()
# # #     print("-" * 60)
# # #     print("STEP 1: WASTE GATE")
# # #     print("-" * 60)

# # #     (
# # #         is_waste,
# # #         waste_probability,
# # #         non_waste_probability
# # #     ) = check_waste(
# # #         waste_gate,
# # #         image
# # #     )

# # #     print(
# # #         f"Waste probability: "
# # #         f"{waste_probability * 100:.2f}%"
# # #     )

# # #     print(
# # #         f"Non-waste probability: "
# # #         f"{non_waste_probability * 100:.2f}%"
# # #     )


# # #     if not is_waste:

# # #         print()
# # #         print(
# # #             "Result: NON-WASTE"
# # #         )

# # #         print(
# # #             "Please upload an image "
# # #             "containing visible waste."
# # #         )

# # #         print()
# # #         print("=" * 60)

# # #         return


# # #     print()
# # #     print(
# # #         "Result: WASTE DETECTED"
# # #     )


# # #     # ========================================================
# # #     # STEP 2: WASTE TYPE DETECTION
# # #     # ========================================================

# # #     print()
# # #     print("-" * 60)
# # #     print(
# # #         "STEP 2: WASTE TYPE DETECTION"
# # #     )
# # #     print("-" * 60)

# # #     (
# # #         detections,
# # #         image_shape
# # #     ) = analyze_waste(
# # #         yolo_model,
# # #         image_path
# # #     )


# # #     # ========================================================
# # #     # CASE A: YOLO FOUND WASTE
# # #     # ========================================================

# # #     if detections:

# # #         print()

# # #         print(
# # #             f"Waste objects detected: "
# # #             f"{len(detections)}"
# # #         )

# # #         print()

# # #         for index, detection in enumerate(
# # #             detections,
# # #             start=1
# # #         ):

# # #             print(
# # #                 f"{index}. "
# # #                 f"{detection['type']} "
# # #                 f"("
# # #                 f"{detection['confidence'] * 100:.2f}%"
# # #                 f")"
# # #             )


# # #         # ----------------------------------------------------
# # #         # STEP 3: AMOUNT
# # #         # ----------------------------------------------------

# # #         image_height = (
# # #             image_shape[0]
# # #         )

# # #         image_width = (
# # #             image_shape[1]
# # #         )

# # #         coverage = (
# # #             calculate_waste_coverage(
# # #                 detections,
# # #                 image_width,
# # #                 image_height
# # #             )
# # #         )

# # #         amount = (
# # #             estimate_waste_amount(
# # #                 coverage,
# # #                 detections
# # #             )
# # #         )


# # #         print()
# # #         print("-" * 60)
# # #         print(
# # #             "STEP 3: VISIBLE WASTE AMOUNT ESTIMATION"
# # #         )
# # #         print("-" * 60)

# # #         print()

# # #         print(
# # #             f"Image dimensions: "
# # #             f"{image_width} x "
# # #             f"{image_height}"
# # #         )

# # #         print(
# # #             f"Visible waste coverage: "
# # #             f"{coverage:.2f}%"
# # #         )

# # #         print(
# # #             f"Estimated amount: "
# # #             f"{amount}"
# # #         )


# # #         # ----------------------------------------------------
# # #         # STEP 4: VEHICLE
# # #         # ----------------------------------------------------

# # #         vehicle = (
# # #             recommend_vehicle(
# # #                 amount
# # #             )
# # #         )


# # #         print()
# # #         print("-" * 60)
# # #         print(
# # #             "STEP 4: VEHICLE RECOMMENDATION"
# # #         )
# # #         print("-" * 60)

# # #         print()

# # #         print(
# # #             "Recommended vehicle:"
# # #         )

# # #         print(
# # #             f"  {vehicle['vehicle']}"
# # #         )

# # #         print()

# # #         print(
# # #             "Reason:"
# # #         )

# # #         print(
# # #             f"  {vehicle['reason']}"
# # #         )


# # #         # ----------------------------------------------------
# # #         # FINAL
# # #         # ----------------------------------------------------

# # #         print()
# # #         print("=" * 60)
# # #         print(
# # #             "SWACHHLENS FINAL RESULT"
# # #         )
# # #         print("=" * 60)

# # #         print()

# # #         print(
# # #             "Waste detected: YES"
# # #         )

# # #         print(
# # #             f"Waste confidence: "
# # #             f"{waste_probability * 100:.2f}%"
# # #         )

# # #         print(
# # #             f"Waste objects detected: "
# # #             f"{len(detections)}"
# # #         )

# # #         print(
# # #             f"Visible waste coverage: "
# # #             f"{coverage:.2f}%"
# # #         )

# # #         print(
# # #             f"Estimated amount: "
# # #             f"{amount}"
# # #         )

# # #         print(
# # #             f"Recommended vehicle: "
# # #             f"{vehicle['vehicle']}"
# # #         )

# # #         print()
# # #         print("=" * 60)

# # #         return


# # #     # ========================================================
# # #     # CASE B: YOLO FOUND NO SPECIFIC OBJECT
# # #     # ========================================================

# # #     print()

# # #     print(
# # #         "No specific waste object "
# # #         "was detected by YOLO."
# # #     )

# # #     print(
# # #         f"No detection reached the "
# # #         f"{YOLO_CONFIDENCE * 100:.0f}% "
# # #         "confidence threshold."
# # #     )


# # #     # ========================================================
# # #     # STEP 2B: SCENE ANALYSIS
# # #     # ========================================================

# # #     print()
# # #     print("-" * 60)
# # #     print(
# # #         "STEP 2B: SCENE ANALYSIS"
# # #     )
# # #     print("-" * 60)


# # #     (
# # #         scene,
# # #         scene_confidence,
# # #         scene_probabilities
# # #     ) = analyze_scene(
# # #         scene_feature_extractor,
# # #         scene_classifier,
# # #         scene_classes,
# # #         image
# # #     )


# # #     print()

# # #     print(
# # #         f"Detected scene: "
# # #         f"{scene}"
# # #     )

# # #     print(
# # #         f"Scene confidence: "
# # #         f"{scene_confidence * 100:.2f}%"
# # #     )


# # #     # ========================================================
# # #     # SCENE CONFIDENCE
# # #     # ========================================================

# # #     if scene_confidence < SCENE_CONFIDENCE:

# # #         print()

# # #         print(
# # #             "Scene classification confidence "
# # #             "is low."
# # #         )

# # #         print(
# # #             "The image contains waste, but "
# # #             "the scene type is uncertain."
# # #         )

# # #         print()

# # #         print(
# # #             "Amount estimation: "
# # #             "MEDIUM (fallback estimate)"
# # #         )

# # #         amount = "MEDIUM"

# # #     else:

# # #         # ----------------------------------------------------
# # #         # SCENE-BASED AMOUNT
# # #         # ----------------------------------------------------

# # #         amount = (
# # #             estimate_scene_amount(
# # #                 scene
# # #             )
# # #         )

# # #         print()

# # #         print(
# # #             f"Scene-based amount estimation: "
# # #             f"{amount}"
# # #         )


# # #     # ========================================================
# # #     # STEP 4: VEHICLE
# # #     # ========================================================

# # #     vehicle = (
# # #         recommend_vehicle(
# # #             amount
# # #         )
# # #     )


# # #     print()
# # #     print("-" * 60)
# # #     print(
# # #         "STEP 3: WASTE AMOUNT ESTIMATION"
# # #     )
# # #     print("-" * 60)

# # #     print()

# # #     print(
# # #         "Specific waste objects were "
# # #         "not detected."
# # #     )

# # #     print(
# # #         f"Scene type: "
# # #         f"{scene}"
# # #     )

# # #     print(
# # #         f"Estimated amount: "
# # #         f"{amount}"
# # #     )


# # #     print()
# # #     print("-" * 60)
# # #     print(
# # #         "STEP 4: VEHICLE RECOMMENDATION"
# # #     )
# # #     print("-" * 60)

# # #     print()

# # #     print(
# # #         "Recommended vehicle:"
# # #     )

# # #     print(
# # #         f"  {vehicle['vehicle']}"
# # #     )

# # #     print()

# # #     print(
# # #         "Reason:"
# # #     )

# # #     print(
# # #         f"  {vehicle['reason']}"
# # #     )


# # #     # ========================================================
# # #     # FINAL RESULT
# # #     # ========================================================

# # #     print()
# # #     print("=" * 60)
# # #     print(
# # #         "SWACHHLENS FINAL RESULT"
# # #     )
# # #     print("=" * 60)

# # #     print()

# # #     print(
# # #         "Waste detected: YES"
# # #     )

# # #     print(
# # #         f"Waste confidence: "
# # #         f"{waste_probability * 100:.2f}%"
# # #     )

# # #     print(
# # #         "Waste objects detected: 0"
# # #     )

# # #     print(
# # #         f"Scene classification: "
# # #         f"{scene}"
# # #     )

# # #     print(
# # #         f"Scene confidence: "
# # #         f"{scene_confidence * 100:.2f}%"
# # #     )

# # #     print(
# # #         f"Estimated amount: "
# # #         f"{amount}"
# # #     )

# # #     print(
# # #         f"Recommended vehicle: "
# # #         f"{vehicle['vehicle']}"
# # #     )

# # #     print()

# # #     print(
# # #         "Note: Scene-based amount is an "
# # #         "estimated accumulation level, "
# # #         "not an exact physical volume."
# # #     )

# # #     print()
# # #     print("=" * 60)


# # # # ============================================================
# # # # COMMAND LINE
# # # # ============================================================

# # # if __name__ == "__main__":

# # #     if len(sys.argv) != 2:

# # #         print(
# # #             "Usage:"
# # #         )

# # #         print(
# # #             'python ai\\src\\analyze_image.py '
# # #             '"path\\to\\image.jpg"'
# # #         )

# # #         sys.exit(1)

# # #     analyze_image(
# # #         sys.argv[1]
# # #     )

# # import io
# # from pathlib import Path
# # import sys

# # import torch
# # import torch.nn as nn
# # from torchvision import transforms, models
# # from torchvision.models import mobilenet_v3_small
# # from PIL import Image
# # from ultralytics import YOLO


# # BASE_DIR = Path(__file__).resolve().parents[1]

# # WASTE_GATE_MODEL = BASE_DIR / "models" / "waste_gate_v2_best.pth"

# # YOLO_MODEL = (
# #     BASE_DIR 
# #     / "models"
# #     / "waste_types_best.pt"
# # )

# # SCENE_MODEL = BASE_DIR / "models" / "scene_analysis_resnet18_best.pth"

# # WASTE_THRESHOLD = 0.40
# # STRONG_WASTE_THRESHOLD = 0.65
# # YOLO_CONFIDENCE = 0.50
# # SCENE_CONFIDENCE = 0.70

# # DEVICE = torch.device(
# #     "cuda" if torch.cuda.is_available() else "cpu"
# # )

# # SCENE_CLASSES = [
# #     "construction_debris",
# #     "drain_blockage",
# #     "garbage_dump",
# #     "overflowing_bin",
# # ]

# # scene_transform = transforms.Compose([
# #     transforms.Resize((224, 224)),
# #     transforms.ToTensor(),
# #     transforms.Normalize(
# #         mean=[0.485, 0.456, 0.406],
# #         std=[0.229, 0.224, 0.225],
# #     ),
# # ])

# # waste_gate_transform = transforms.Compose([
# #     transforms.Resize((224, 224)),
# #     transforms.ToTensor(),
# #     transforms.Normalize(
# #         mean=[0.485, 0.456, 0.406],
# #         std=[0.229, 0.224, 0.225],
# #     ),
# # ])


# # def load_waste_gate():
# #     print("Loading Waste Gate V2...")

# #     model = mobilenet_v3_small(
# #         weights=None,
# #         num_classes=2,
# #     )

# #     checkpoint = torch.load(
# #         WASTE_GATE_MODEL,
# #         map_location=DEVICE,
# #     )

# #     if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
# #         state_dict = checkpoint["model_state_dict"]
# #     elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
# #         state_dict = checkpoint["state_dict"]
# #     else:
# #         state_dict = checkpoint

# #     model.load_state_dict(state_dict)
# #     model.to(DEVICE)
# #     model.eval()

# #     return model


# # def load_yolo():
# #     print("Loading YOLO waste detector...")
# #     return YOLO(str(YOLO_MODEL))


# # def load_scene_model():
# #     print("Loading scene analysis model...")

# #     checkpoint = torch.load(
# #         SCENE_MODEL,
# #         map_location=DEVICE,
# #     )

# #     classes = checkpoint.get("classes", SCENE_CLASSES)

# #     resnet = models.resnet18(weights=None)

# #     feature_extractor = nn.Sequential(
# #         *list(resnet.children())[:-1]
# #     )
# #     feature_extractor.to(DEVICE)
# #     feature_extractor.eval()

# #     classifier = nn.Linear(
# #         resnet.fc.in_features,
# #         len(classes),
# #     )

# #     classifier.load_state_dict(
# #         checkpoint["classifier_state_dict"]
# #     )
# #     classifier.to(DEVICE)
# #     classifier.eval()

# #     return feature_extractor, classifier, classes


# # def check_waste(model, image):
# #     image_tensor = (
# #         waste_gate_transform(image)
# #         .unsqueeze(0)
# #         .to(DEVICE)
# #     )

# #     with torch.no_grad():
# #         output = model(image_tensor)
# #         probabilities = torch.softmax(output, dim=1)[0]

# #     non_waste_probability = probabilities[0].item()
# #     waste_probability = probabilities[1].item()

# #     return (
# #         waste_probability >= WASTE_THRESHOLD,
# #         waste_probability,
# #         non_waste_probability,
# #     )


# # # def analyze_waste(yolo_model, image_path):
# #     # results = yolo_model.predict(
# #         # source=str(image_path),
# #         # conf=YOLO_CONFIDENCE,
# #         # verbose=False,
# #     # )
# # # 
# #     # result = results[0]
# #     # detections = []
# # # 
# #     # if result.boxes is None:
# #         # return detections, result.orig_shape
# # # 
# #     # for box in result.boxes:
# #         # class_id = int(box.cls[0].item())
# #         # confidence = float(box.conf[0].item())
# #         # xyxy = box.xyxy[0].tolist()
# # # 
# #         # detections.append({
# #             # "type": result.names[class_id],
# #             # "confidence": confidence,
# #             # "box": xyxy,
# #         # })
# # # 
# #     # return detections, result.orig_shape

# # def analyze_waste(yolo_model, image):
# #     results = yolo_model.predict(
# #         source=image,
# #         conf=YOLO_CONFIDENCE,
# #         verbose=False,
# #     )

# #     result = results[0]
# #     detections = []

# #     if result.boxes is None:
# #         return detections, result.orig_shape

# #     for box in result.boxes:
# #         class_id = int(box.cls[0].item())
# #         confidence = float(box.conf[0].item())
# #         xyxy = box.xyxy[0].tolist()

# #         detections.append({
# #             "type": result.names[class_id],
# #             "confidence": confidence,
# #             "box": xyxy,
# #         })

# #     return detections, result.orig_shape

# # def analyze_scene(feature_extractor, classifier, classes, image):
# #     image_tensor = (
# #         scene_transform(image)
# #         .unsqueeze(0)
# #         .to(DEVICE)
# #     )

# #     with torch.no_grad():
# #         features = feature_extractor(image_tensor)
# #         features = features.view(features.size(0), -1)
# #         outputs = classifier(features)
# #         probabilities = torch.softmax(outputs, dim=1)[0]

# #         predicted_index = int(
# #             torch.argmax(probabilities).item()
# #         )

# #     scene = classes[predicted_index]
# #     confidence = probabilities[predicted_index].item()

# #     return scene, confidence


# # def estimate_scene_amount(scene):
# #     scene_amounts = {
# #         "garbage_dump": "VERY LARGE",
# #         "overflowing_bin": "LARGE",
# #         "construction_debris": "LARGE",
# #         "drain_blockage": "MEDIUM",
# #     }

# #     return scene_amounts.get(scene, "MEDIUM")


# # def recommend_vehicle(amount):
# #     recommendations = {
# #         "SMALL": {
# #             "vehicle": "Small waste collection vehicle",
# #             "reason": "Low visible waste accumulation.",
# #         },
# #         "MEDIUM": {
# #             "vehicle": "Medium waste collection vehicle",
# #             "reason": "Moderate visible waste accumulation.",
# #         },
# #         "LARGE": {
# #             "vehicle": "Large waste collection vehicle",
# #             "reason": "Large visible waste accumulation.",
# #         },
# #         "VERY LARGE": {
# #             "vehicle": "Large-capacity waste truck",
# #             "reason": (
# #                 "Very large waste accumulation requires "
# #                 "a high-capacity vehicle."
# #             ),
# #         },
# #     }

# #     return recommendations.get(
# #         amount,
# #         {
# #             "vehicle": "Vehicle recommendation unavailable",
# #             "reason": "Unknown waste amount category.",
# #         },
# #     )


# # def calculate_union_area(boxes, image_width, image_height):
# #     if not boxes:
# #         return 0.0

# #     rectangles = []

# #     for box in boxes:
# #         x1, y1, x2, y2 = box

# #         x1 = max(0, min(x1, image_width))
# #         y1 = max(0, min(y1, image_height))
# #         x2 = max(0, min(x2, image_width))
# #         y2 = max(0, min(y2, image_height))

# #         if x2 > x1 and y2 > y1:
# #             rectangles.append((x1, y1, x2, y2))

# #     if not rectangles:
# #         return 0.0

# #     x_values = sorted({
# #         value
# #         for x1, _, x2, _ in rectangles
# #         for value in (x1, x2)
# #     })

# #     total_area = 0.0

# #     for i in range(len(x_values) - 1):
# #         left = x_values[i]
# #         right = x_values[i + 1]
# #         width = right - left

# #         if width <= 0:
# #             continue

# #         active_intervals = [
# #             (y1, y2)
# #             for x1, y1, x2, y2 in rectangles
# #             if x1 < right and x2 > left
# #         ]

# #         if not active_intervals:
# #             continue

# #         active_intervals.sort()

# #         current_start, current_end = active_intervals[0]
# #         covered_height = 0.0

# #         for start, end in active_intervals[1:]:
# #             if start <= current_end:
# #                 current_end = max(current_end, end)
# #             else:
# #                 covered_height += current_end - current_start
# #                 current_start = start
# #                 current_end = end

# #         covered_height += current_end - current_start
# #         total_area += width * covered_height

# #     return total_area


# # def calculate_waste_coverage(detections, image_width, image_height):
# #     boxes = [detection["box"] for detection in detections]

# #     waste_area = calculate_union_area(
# #         boxes,
# #         image_width,
# #         image_height,
# #     )

# #     total_image_area = image_width * image_height

# #     if total_image_area <= 0:
# #         return 0.0

# #     return (waste_area / total_image_area) * 100


# # def estimate_waste_amount(coverage, detections):
# #     object_count = len(detections)

# #     if object_count == 1:
# #         return "SMALL"

# #     if object_count <= 3:
# #         return "SMALL" if coverage < 25 else "MEDIUM"

# #     if object_count <= 7:
# #         return "MEDIUM" if coverage < 20 else "LARGE"

# #     if object_count <= 15:
# #         return "LARGE" if coverage < 20 else "VERY LARGE"

# #     return "VERY LARGE"


# # def analyze_image(image_path):
# #     image_path = Path(image_path)

# #     if not image_path.exists():
# #         print(f"ERROR: Image not found: {image_path}")
# #         return

# #     print("\n" + "=" * 60)
# #     print("SWACHHLENS IMAGE ANALYSIS")
# #     print("=" * 60)
# #     print(f"Image: {image_path.name}")
# #     print(f"Device: {DEVICE}")

# #     try:
# #         image = Image.open(image_path).convert("RGB")
# #     except Exception as error:
# #         print(f"ERROR: Could not open image: {error}")
# #         return

# #     waste_gate = load_waste_gate()
# #     yolo_model = load_yolo()
# #     scene_feature_extractor, scene_classifier, scene_classes = (
# #         load_scene_model()
# #     )

# #     print("\n" + "-" * 60)
# #     print("STEP 1: WASTE GATE")
# #     print("-" * 60)

# #     is_waste, waste_probability, non_waste_probability = check_waste(
# #         waste_gate,
# #         image,
# #     )

# #     print(f"Waste probability: {waste_probability * 100:.2f}%")
# #     print(
# #         f"Non-waste probability: "
# #         f"{non_waste_probability * 100:.2f}%"
# #     )

# #     if not is_waste:
# #         print("\nResult: NON-WASTE")
# #         print("Please upload an image containing visible waste.")
# #         return

# #     print("\nResult: WASTE DETECTED")

# #     print("\n" + "-" * 60)
# #     print("STEP 2: WASTE TYPE DETECTION")
# #     print("-" * 60)

# #     detections, image_shape = analyze_waste(
# #         yolo_model,
# #         image_path,
# #     )

# #     if detections:
# #         print(f"\nWaste objects detected: {len(detections)}")

# #         for index, detection in enumerate(detections, start=1):
# #             print(
# #                 f"{index}. {detection['type']} "
# #                 f"({detection['confidence'] * 100:.2f}%)"
# #             )

# #         image_height, image_width = image_shape

# #         coverage = calculate_waste_coverage(
# #             detections,
# #             image_width,
# #             image_height,
# #         )

# #         amount = estimate_waste_amount(
# #             coverage,
# #             detections,
# #         )

# #         vehicle = recommend_vehicle(amount)

# #         print("\n" + "-" * 60)
# #         print("STEP 3: VISIBLE WASTE AMOUNT ESTIMATION")
# #         print("-" * 60)
# #         print(f"Image dimensions: {image_width} x {image_height}")
# #         print(f"Visible waste coverage: {coverage:.2f}%")
# #         print(f"Estimated amount: {amount}")

# #         print("\n" + "-" * 60)
# #         print("STEP 4: VEHICLE RECOMMENDATION")
# #         print("-" * 60)
# #         print(f"Recommended vehicle: {vehicle['vehicle']}")
# #         print(f"Reason: {vehicle['reason']}")

# #         print("\n" + "=" * 60)
# #         print("SWACHHLENS FINAL RESULT")
# #         print("=" * 60)
# #         print("Waste detected: YES")
# #         print(
# #             f"Waste confidence: "
# #             f"{waste_probability * 100:.2f}%"
# #         )
# #         print(f"Waste objects detected: {len(detections)}")
# #         print(f"Visible waste coverage: {coverage:.2f}%")
# #         print(f"Estimated amount: {amount}")
# #         print(f"Recommended vehicle: {vehicle['vehicle']}")
# #         print("=" * 60)

# #         return

# #     print("\nNo specific waste object was detected by YOLO.")
# #     print(
# #         f"No detection reached the "
# #         f"{YOLO_CONFIDENCE * 100:.0f}% confidence threshold."
# #     )

# #     print("\n" + "-" * 60)
# #     print("STEP 2B: SCENE ANALYSIS")
# #     print("-" * 60)

# #     scene, scene_confidence = analyze_scene(
# #         scene_feature_extractor,
# #         scene_classifier,
# #         scene_classes,
# #         image,
# #     )

# #     print(f"Detected scene: {scene}")
# #     print(f"Scene confidence: {scene_confidence * 100:.2f}%")

# #     if scene_confidence < SCENE_CONFIDENCE:
# #         amount = "MEDIUM"
# #         print("\nScene classification confidence is low.")
# #         print("Amount estimation: MEDIUM (fallback estimate)")
# #     else:
# #         amount = estimate_scene_amount(scene)
# #         print(f"Scene-based amount estimation: {amount}")

# #     vehicle = recommend_vehicle(amount)

# #     print("\n" + "-" * 60)
# #     print("STEP 3: WASTE AMOUNT ESTIMATION")
# #     print("-" * 60)
# #     print("Specific waste objects were not detected.")
# #     print(f"Scene type: {scene}")
# #     print(f"Estimated amount: {amount}")

# #     print("\n" + "-" * 60)
# #     print("STEP 4: VEHICLE RECOMMENDATION")
# #     print("-" * 60)
# #     print(f"Recommended vehicle: {vehicle['vehicle']}")
# #     print(f"Reason: {vehicle['reason']}")

# #     print("\n" + "=" * 60)
# #     print("SWACHHLENS FINAL RESULT")
# #     print("=" * 60)
# #     print("Waste detected: YES")
# #     print(
# #         f"Waste confidence: "
# #         f"{waste_probability * 100:.2f}%"
# #     )
# #     print("Waste objects detected: 0")
# #     print(f"Scene classification: {scene}")
# #     print(
# #         f"Scene confidence: "
# #         f"{scene_confidence * 100:.2f}%"
# #     )
# #     print(f"Estimated amount: {amount}")
# #     print(f"Recommended vehicle: {vehicle['vehicle']}")
# #     print(
# #         "\nNote: Scene-based amount is an estimated "
# #         "accumulation level, not an exact physical volume."
# #     )
# #     print("=" * 60)

# # def analyze_image_bytes(image_bytes):
# #     """
# #     Backend-friendly entry point.

# #     Uses:
# #         1. Waste Gate V2
# #         2. YOLO waste detector
# #         3. Scene Analysis ResNet18

# #     The Waste Gate is the first filter.
# #     YOLO confirms specific waste objects.
# #     Scene Analysis is only used for strong waste-scene cases.
# #     """

# #     image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

# #     waste_gate = load_waste_gate()
# #     yolo_model = load_yolo()
# #     scene_feature_extractor, scene_classifier, scene_classes = load_scene_model()

# #     # ---------------------------------------------------------
# #     # STEP 1: WASTE GATE
# #     # ---------------------------------------------------------
# #     is_waste, waste_probability, non_waste_probability = check_waste(
# #         waste_gate,
# #         image,
# #     )

# #     # Clearly non-waste
# #     if waste_probability < WASTE_THRESHOLD:
# #         return {
# #             "valid": False,
# #             "reason": "Image does not appear to contain waste.",
# #             "wasteType": None,
# #             "severity": None,
# #             "confidence": int(round(non_waste_probability * 100)),
# #             "engine": "ai",
# #             "details": [],
# #             "summary": "Image rejected by the waste gate.",
# #         }

# #     # ---------------------------------------------------------
# #     # STEP 2: YOLO WASTE DETECTION
# #     # ---------------------------------------------------------
# #     detections, original_shape = analyze_waste(
# #         yolo_model,
# #         image,
# #     )

# #     # ---------------------------------------------------------
# #     # STEP 3: YOLO CONFIRMED WASTE
# #     # ---------------------------------------------------------
# #     if detections:

# #         type_mapping = {
# #             "hazardous-waste": "Hazardous",
# #             "medical-waste": "Hazardous",
# #             "organic-waste": "Organic",

# #             "recyclable-waste-cardboard": "Plastic",
# #             "recyclable-waste-clothes": "Plastic",
# #             "recyclable-waste-glass": "Plastic",
# #             "recyclable-waste-metal": "Plastic",
# #             "recyclable-waste-nylonbag": "Plastic",
# #             "recyclable-waste-paper": "Plastic",
# #             "recyclable-waste-paperbag": "Plastic",
# #             "recyclable-waste-plastic": "Plastic",
# #             "recyclable-waste-shoe": "Plastic",
# #         }

# #         type_counts = {}

# #         for detection in detections:
# #             model_type = detection["type"]

# #             app_type = type_mapping.get(
# #                 model_type,
# #                 "Plastic",
# #             )

# #             type_counts[app_type] = (
# #                 type_counts.get(app_type, 0) + 1
# #             )

# #         waste_type = max(
# #             type_counts,
# #             key=type_counts.get,
# #         )

# #         highest_confidence = max(
# #             detection["confidence"]
# #             for detection in detections
# #         )

# #         # Require the strongest YOLO detection to pass
# #         # the stricter confidence threshold.
# #         if highest_confidence < YOLO_CONFIDENCE:

# #             # If the Waste Gate is not strongly positive,
# #             # don't trust the weak YOLO detection.
# #             if waste_probability < STRONG_WASTE_THRESHOLD:
# #                 return {
# #                     "valid": False,
# #                     "reason": (
# #                         "Waste could not be identified "
# #                         "with sufficient confidence."
# #                     ),
# #                     "wasteType": None,
# #                     "severity": None,
# #                     "confidence": int(
# #                         round(waste_probability * 100)
# #                     ),
# #                     "engine": "ai",
# #                     "details": [],
# #                     "summary": (
# #                         "The AI models were not confident "
# #                         "enough to identify waste."
# #                     ),
# #                 }

# #         # Severity based on number of detected objects.
# #         if len(detections) >= 5:
# #             severity = "High"
# #         elif len(detections) >= 2:
# #             severity = "Medium"
# #         else:
# #             severity = "Low"

# #         details = [
# #             {
# #                 "label": detection["type"],
# #                 "count": 1,
# #                 "conf": int(
# #                     round(detection["confidence"] * 100)
# #                 ),
# #             }
# #             for detection in detections
# #         ]

# #         summary = (
# #             f"{len(detections)} waste item"
# #             f"{'s' if len(detections) != 1 else ''} detected. "
# #             f"Primary type: {waste_type}."
# #         )

# #         return {
# #             "valid": True,
# #             "reason": None,
# #             "wasteType": waste_type,
# #             "severity": severity,
# #             "confidence": int(
# #                 round(highest_confidence * 100)
# #             ),
# #             "engine": "ai",
# #             "details": details,
# #             "summary": summary,
# #         }

# #     # ---------------------------------------------------------
# #     # STEP 4: NO YOLO OBJECT DETECTED
# #     # ---------------------------------------------------------

# #     # A weak/moderate Waste Gate result without a YOLO
# #     # detection is not enough to call something waste.
# #     if waste_probability < STRONG_WASTE_THRESHOLD:
# #         return {
# #             "valid": False,
# #             "reason": (
# #                 "Waste could not be confirmed in the image."
# #             ),
# #             "wasteType": None,
# #             "severity": None,
# #             "confidence": int(
# #                 round(waste_probability * 100)
# #             ),
# #             "engine": "ai",
# #             "details": [],
# #             "summary": (
# #                 "The waste gate detected possible waste, "
# #                 "but no waste object was confirmed."
# #             ),
# #         }

# #     # ---------------------------------------------------------
# #     # STEP 5: STRONG WASTE SCENE
# #     # ---------------------------------------------------------
# #     scene, scene_confidence = analyze_scene(
# #         scene_feature_extractor,
# #         scene_classifier,
# #         scene_classes,
# #         image,
# #     )

# #     # Scene model must also be confident.
# #     if scene_confidence < SCENE_CONFIDENCE:
# #         return {
# #             "valid": False,
# #             "reason": (
# #                 "Waste scene could not be identified "
# #                 "with sufficient confidence."
# #             ),
# #             "wasteType": None,
# #             "severity": None,
# #             "confidence": int(
# #                 round(scene_confidence * 100)
# #             ),
# #             "engine": "ai",
# #             "details": [],
# #             "summary": (
# #                 "The AI could not confidently identify "
# #                 "the waste scene."
# #             ),
# #         }

# #     amount = estimate_scene_amount(scene)

# #     amount_to_severity = {
# #         "SMALL": "Low",
# #         "MEDIUM": "Medium",
# #         "LARGE": "High",
# #         "VERY LARGE": "High",
# #     }

# #     severity = amount_to_severity.get(
# #         amount,
# #         "Medium",
# #     )

# #     vehicle = recommend_vehicle(amount)

# #     return {
# #         "valid": True,
# #         "reason": None,
# #         "wasteType": None,
# #         "severity": severity,
# #         "confidence": int(
# #             round(scene_confidence * 100)
# #         ),
# #         "engine": "ai",
# #         "details": [
# #             {
# #                 "label": scene,
# #                 "count": 1,
# #                 "conf": int(
# #                     round(scene_confidence * 100)
# #                 ),
# #             }
# #         ],
# #         "summary": (
# #             f"Scene classified as {scene}. "
# #             f"Estimated waste amount: {amount}. "
# #             f"Recommended vehicle: "
# #             f"{vehicle['vehicle']}."
# #         ),
# #     }
# # if __name__ == "__main__":
# #     if len(sys.argv) != 2:
# #         print("Usage:")
# #         print(
# #             'python ai\\src\\analyze_image.py '
# #             '"path\\to\\image.jpg"'
# #         )
# #         sys.exit(1)

# #     analyze_image(sys.argv[1])


# import io
# from pathlib import Path
# import sys

# import torch
# import torch.nn as nn
# from torchvision import transforms, models
# from torchvision.models import mobilenet_v3_small
# from PIL import Image
# from ultralytics import YOLO


# BASE_DIR = Path(__file__).resolve().parents[1]

# WASTE_GATE_MODEL = BASE_DIR / "models" / "waste_gate_v2_best.pth"

# YOLO_MODEL = (
#     BASE_DIR 
#     / "models"
#     / "waste_types_best.pt"
# )

# SCENE_MODEL = BASE_DIR / "models" / "scene_analysis_resnet18_best.pth"

# WASTE_THRESHOLD = 0.40
# STRONG_WASTE_THRESHOLD = 0.65
# YOLO_CONFIDENCE = 0.50
# YOLO_STRONG_THRESHOLD = 0.60
# SCENE_CONFIDENCE = 0.70

# DEVICE = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )

# SCENE_CLASSES = [
#     "construction_debris",
#     "drain_blockage",
#     "garbage_dump",
#     "overflowing_bin",
# ]

# scene_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],
#     ),
# ])

# waste_gate_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],
#     ),
# ])


# def load_waste_gate():
#     print("Loading Waste Gate V2...")

#     model = mobilenet_v3_small(
#         weights=None,
#         num_classes=2,
#     )

#     checkpoint = torch.load(
#         WASTE_GATE_MODEL,
#         map_location=DEVICE,
#     )

#     if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
#         state_dict = checkpoint["model_state_dict"]
#     elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
#         state_dict = checkpoint["state_dict"]
#     else:
#         state_dict = checkpoint

#     model.load_state_dict(state_dict)
#     model.to(DEVICE)
#     model.eval()

#     return model


# def load_yolo():
#     print("Loading YOLO waste detector...")
#     return YOLO(str(YOLO_MODEL))


# def load_scene_model():
#     print("Loading scene analysis model...")

#     checkpoint = torch.load(
#         SCENE_MODEL,
#         map_location=DEVICE,
#     )

#     classes = checkpoint.get("classes", SCENE_CLASSES)

#     resnet = models.resnet18(weights=None)

#     feature_extractor = nn.Sequential(
#         *list(resnet.children())[:-1]
#     )
#     feature_extractor.to(DEVICE)
#     feature_extractor.eval()

#     classifier = nn.Linear(
#         resnet.fc.in_features,
#         len(classes),
#     )

#     classifier.load_state_dict(
#         checkpoint["classifier_state_dict"]
#     )
#     classifier.to(DEVICE)
#     classifier.eval()

#     return feature_extractor, classifier, classes


# def check_waste(model, image):
#     image_tensor = (
#         waste_gate_transform(image)
#         .unsqueeze(0)
#         .to(DEVICE)
#     )

#     with torch.no_grad():
#         output = model(image_tensor)
#         probabilities = torch.softmax(output, dim=1)[0]

#     non_waste_probability = probabilities[0].item()
#     waste_probability = probabilities[1].item()

#     return (
#         waste_probability >= WASTE_THRESHOLD,
#         waste_probability,
#         non_waste_probability,
#     )


# # def analyze_waste(yolo_model, image_path):
#     # results = yolo_model.predict(
#         # source=str(image_path),
#         # conf=YOLO_CONFIDENCE,
#         # verbose=False,
#     # )
# # 
#     # result = results[0]
#     # detections = []
# # 
#     # if result.boxes is None:
#         # return detections, result.orig_shape
# # 
#     # for box in result.boxes:
#         # class_id = int(box.cls[0].item())
#         # confidence = float(box.conf[0].item())
#         # xyxy = box.xyxy[0].tolist()
# # 
#         # detections.append({
#             # "type": result.names[class_id],
#             # "confidence": confidence,
#             # "box": xyxy,
#         # })
# # 
#     # return detections, result.orig_shape

# def analyze_waste(yolo_model, image):
#     results = yolo_model.predict(
#         source=image,
#         conf=YOLO_CONFIDENCE,
#         verbose=False,
#     )

#     result = results[0]
#     detections = []

#     if result.boxes is None:
#         return detections, result.orig_shape

#     for box in result.boxes:
#         class_id = int(box.cls[0].item())
#         confidence = float(box.conf[0].item())
#         xyxy = box.xyxy[0].tolist()

#         detections.append({
#             "type": result.names[class_id],
#             "confidence": confidence,
#             "box": xyxy,
#         })

#     return detections, result.orig_shape

# def analyze_scene(feature_extractor, classifier, classes, image):
#     image_tensor = (
#         scene_transform(image)
#         .unsqueeze(0)
#         .to(DEVICE)
#     )

#     with torch.no_grad():
#         features = feature_extractor(image_tensor)
#         features = features.view(features.size(0), -1)
#         outputs = classifier(features)
#         probabilities = torch.softmax(outputs, dim=1)[0]

#         predicted_index = int(
#             torch.argmax(probabilities).item()
#         )

#     scene = classes[predicted_index]
#     confidence = probabilities[predicted_index].item()

#     return scene, confidence


# def estimate_scene_amount(scene):
#     scene_amounts = {
#         "garbage_dump": "VERY LARGE",
#         "overflowing_bin": "LARGE",
#         "construction_debris": "LARGE",
#         "drain_blockage": "MEDIUM",
#     }

#     return scene_amounts.get(scene, "MEDIUM")


# def recommend_vehicle(amount):
#     recommendations = {
#         "SMALL": {
#             "vehicle": "Small waste collection vehicle",
#             "reason": "Low visible waste accumulation.",
#         },
#         "MEDIUM": {
#             "vehicle": "Medium waste collection vehicle",
#             "reason": "Moderate visible waste accumulation.",
#         },
#         "LARGE": {
#             "vehicle": "Large waste collection vehicle",
#             "reason": "Large visible waste accumulation.",
#         },
#         "VERY LARGE": {
#             "vehicle": "Large-capacity waste truck",
#             "reason": (
#                 "Very large waste accumulation requires "
#                 "a high-capacity vehicle."
#             ),
#         },
#     }

#     return recommendations.get(
#         amount,
#         {
#             "vehicle": "Vehicle recommendation unavailable",
#             "reason": "Unknown waste amount category.",
#         },
#     )


# def calculate_union_area(boxes, image_width, image_height):
#     if not boxes:
#         return 0.0

#     rectangles = []

#     for box in boxes:
#         x1, y1, x2, y2 = box

#         x1 = max(0, min(x1, image_width))
#         y1 = max(0, min(y1, image_height))
#         x2 = max(0, min(x2, image_width))
#         y2 = max(0, min(y2, image_height))

#         if x2 > x1 and y2 > y1:
#             rectangles.append((x1, y1, x2, y2))

#     if not rectangles:
#         return 0.0

#     x_values = sorted({
#         value
#         for x1, _, x2, _ in rectangles
#         for value in (x1, x2)
#     })

#     total_area = 0.0

#     for i in range(len(x_values) - 1):
#         left = x_values[i]
#         right = x_values[i + 1]
#         width = right - left

#         if width <= 0:
#             continue

#         active_intervals = [
#             (y1, y2)
#             for x1, y1, x2, y2 in rectangles
#             if x1 < right and x2 > left
#         ]

#         if not active_intervals:
#             continue

#         active_intervals.sort()

#         current_start, current_end = active_intervals[0]
#         covered_height = 0.0

#         for start, end in active_intervals[1:]:
#             if start <= current_end:
#                 current_end = max(current_end, end)
#             else:
#                 covered_height += current_end - current_start
#                 current_start = start
#                 current_end = end

#         covered_height += current_end - current_start
#         total_area += width * covered_height

#     return total_area


# def calculate_waste_coverage(detections, image_width, image_height):
#     boxes = [detection["box"] for detection in detections]

#     waste_area = calculate_union_area(
#         boxes,
#         image_width,
#         image_height,
#     )

#     total_image_area = image_width * image_height

#     if total_image_area <= 0:
#         return 0.0

#     return (waste_area / total_image_area) * 100


# def estimate_waste_amount(coverage, detections):
#     object_count = len(detections)

#     if object_count == 1:
#         return "SMALL"

#     if object_count <= 3:
#         return "SMALL" if coverage < 25 else "MEDIUM"

#     if object_count <= 7:
#         return "MEDIUM" if coverage < 20 else "LARGE"

#     if object_count <= 15:
#         return "LARGE" if coverage < 20 else "VERY LARGE"

#     return "VERY LARGE"


# def analyze_image(image_path):
#     image_path = Path(image_path)

#     if not image_path.exists():
#         print(f"ERROR: Image not found: {image_path}")
#         return

#     print("\n" + "=" * 60)
#     print("SWACHHLENS IMAGE ANALYSIS")
#     print("=" * 60)
#     print(f"Image: {image_path.name}")
#     print(f"Device: {DEVICE}")

#     try:
#         image = Image.open(image_path).convert("RGB")
#     except Exception as error:
#         print(f"ERROR: Could not open image: {error}")
#         return

#     waste_gate = load_waste_gate()
#     yolo_model = load_yolo()
#     scene_feature_extractor, scene_classifier, scene_classes = (
#         load_scene_model()
#     )

#     print("\n" + "-" * 60)
#     print("STEP 1: WASTE GATE")
#     print("-" * 60)

#     is_waste, waste_probability, non_waste_probability = check_waste(
#         waste_gate,
#         image,
#     )

#     print(f"Waste probability: {waste_probability * 100:.2f}%")
#     print(
#         f"Non-waste probability: "
#         f"{non_waste_probability * 100:.2f}%"
#     )

#     if not is_waste:
#         print("\nResult: NON-WASTE")
#         print("Please upload an image containing visible waste.")
#         return

#     print("\nResult: WASTE DETECTED")

#     print("\n" + "-" * 60)
#     print("STEP 2: WASTE TYPE DETECTION")
#     print("-" * 60)

#     detections, image_shape = analyze_waste(
#         yolo_model,
#         image_path,
#     )

#     if detections:
#         print(f"\nWaste objects detected: {len(detections)}")

#         for index, detection in enumerate(detections, start=1):
#             print(
#                 f"{index}. {detection['type']} "
#                 f"({detection['confidence'] * 100:.2f}%)"
#             )

#         image_height, image_width = image_shape

#         coverage = calculate_waste_coverage(
#             detections,
#             image_width,
#             image_height,
#         )

#         amount = estimate_waste_amount(
#             coverage,
#             detections,
#         )

#         vehicle = recommend_vehicle(amount)

#         print("\n" + "-" * 60)
#         print("STEP 3: VISIBLE WASTE AMOUNT ESTIMATION")
#         print("-" * 60)
#         print(f"Image dimensions: {image_width} x {image_height}")
#         print(f"Visible waste coverage: {coverage:.2f}%")
#         print(f"Estimated amount: {amount}")

#         print("\n" + "-" * 60)
#         print("STEP 4: VEHICLE RECOMMENDATION")
#         print("-" * 60)
#         print(f"Recommended vehicle: {vehicle['vehicle']}")
#         print(f"Reason: {vehicle['reason']}")

#         print("\n" + "=" * 60)
#         print("SWACHHLENS FINAL RESULT")
#         print("=" * 60)
#         print("Waste detected: YES")
#         print(
#             f"Waste confidence: "
#             f"{waste_probability * 100:.2f}%"
#         )
#         print(f"Waste objects detected: {len(detections)}")
#         print(f"Visible waste coverage: {coverage:.2f}%")
#         print(f"Estimated amount: {amount}")
#         print(f"Recommended vehicle: {vehicle['vehicle']}")
#         print("=" * 60)

#         return

#     print("\nNo specific waste object was detected by YOLO.")
#     print(
#         f"No detection reached the "
#         f"{YOLO_CONFIDENCE * 100:.0f}% confidence threshold."
#     )

#     print("\n" + "-" * 60)
#     print("STEP 2B: SCENE ANALYSIS")
#     print("-" * 60)

#     scene, scene_confidence = analyze_scene(
#         scene_feature_extractor,
#         scene_classifier,
#         scene_classes,
#         image,
#     )

#     print(f"Detected scene: {scene}")
#     print(f"Scene confidence: {scene_confidence * 100:.2f}%")

#     if scene_confidence < SCENE_CONFIDENCE:
#         amount = "MEDIUM"
#         print("\nScene classification confidence is low.")
#         print("Amount estimation: MEDIUM (fallback estimate)")
#     else:
#         amount = estimate_scene_amount(scene)
#         print(f"Scene-based amount estimation: {amount}")

#     vehicle = recommend_vehicle(amount)

#     print("\n" + "-" * 60)
#     print("STEP 3: WASTE AMOUNT ESTIMATION")
#     print("-" * 60)
#     print("Specific waste objects were not detected.")
#     print(f"Scene type: {scene}")
#     print(f"Estimated amount: {amount}")

#     print("\n" + "-" * 60)
#     print("STEP 4: VEHICLE RECOMMENDATION")
#     print("-" * 60)
#     print(f"Recommended vehicle: {vehicle['vehicle']}")
#     print(f"Reason: {vehicle['reason']}")

#     print("\n" + "=" * 60)
#     print("SWACHHLENS FINAL RESULT")
#     print("=" * 60)
#     print("Waste detected: YES")
#     print(
#         f"Waste confidence: "
#         f"{waste_probability * 100:.2f}%"
#     )
#     print("Waste objects detected: 0")
#     print(f"Scene classification: {scene}")
#     print(
#         f"Scene confidence: "
#         f"{scene_confidence * 100:.2f}%"
#     )
#     print(f"Estimated amount: {amount}")
#     print(f"Recommended vehicle: {vehicle['vehicle']}")
#     print(
#         "\nNote: Scene-based amount is an estimated "
#         "accumulation level, not an exact physical volume."
#     )
#     print("=" * 60)

# def analyze_image_bytes(image_bytes):
#     """
#     Backend-friendly entry point.

#     Uses:
#         1. Waste Gate V2
#         2. YOLO waste detector
#         3. Scene Analysis ResNet18

#     The Waste Gate is the first filter.
#     YOLO confirms specific waste objects.
#     Scene Analysis is only used for strong waste-scene cases.
#     """

#     image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

#     waste_gate = load_waste_gate()
#     yolo_model = load_yolo()
#     scene_feature_extractor, scene_classifier, scene_classes = load_scene_model()

#     # ---------------------------------------------------------
#     # STEP 1: WASTE GATE
#     # ---------------------------------------------------------
#     is_waste, waste_probability, non_waste_probability = check_waste(
#         waste_gate,
#         image,
#     )

#     # Clearly non-waste
#     if waste_probability < WASTE_THRESHOLD:
#         return {
#             "valid": False,
#             "reason": "Image does not appear to contain waste.",
#             "wasteType": None,
#             "severity": None,
#             "confidence": int(round(non_waste_probability * 100)),
#             "engine": "ai",
#             "details": [],
#             "summary": "Image rejected by the waste gate.",
#         }

#     # ---------------------------------------------------------
#     # STEP 2: YOLO WASTE DETECTION
#     # ---------------------------------------------------------
#     detections, original_shape = analyze_waste(
#         yolo_model,
#         image,
#     )

#     # ---------------------------------------------------------
#     # STEP 3: YOLO CONFIRMED WASTE
#     # ---------------------------------------------------------
#     if detections:

#         type_mapping = {
#             "hazardous-waste": "Hazardous",
#             "medical-waste": "Hazardous",
#             "organic-waste": "Organic",

#             "recyclable-waste-cardboard": "Plastic",
#             "recyclable-waste-clothes": "Plastic",
#             "recyclable-waste-glass": "Plastic",
#             "recyclable-waste-metal": "Plastic",
#             "recyclable-waste-nylonbag": "Plastic",
#             "recyclable-waste-paper": "Plastic",
#             "recyclable-waste-paperbag": "Plastic",
#             "recyclable-waste-plastic": "Plastic",
#             "recyclable-waste-shoe": "Plastic",
#         }

#         type_counts = {}

#         for detection in detections:
#             model_type = detection["type"]

#             app_type = type_mapping.get(
#                 model_type,
#                 "Plastic",
#             )

#             type_counts[app_type] = (
#                 type_counts.get(app_type, 0) + 1
#             )

#         waste_type = max(
#             type_counts,
#             key=type_counts.get,
#         )

#         highest_confidence = max(
#             detection["confidence"]
#             for detection in detections
#         )

#         # Strong YOLO detections are accepted directly.
#         # Weak/moderate detections (20%-50%) are accepted only
#         # when the Waste Gate is strongly positive.
#         yolo_is_strong = (
#             highest_confidence >= YOLO_STRONG_THRESHOLD
#         )

#         yolo_is_acceptable_weak = (
#             highest_confidence >= YOLO_CONFIDENCE
#             and waste_probability >= STRONG_WASTE_THRESHOLD
#         )

#         if not (yolo_is_strong or yolo_is_acceptable_weak):

#             # Do not trust a weak YOLO detection unless the
#             # Waste Gate is strongly positive.
#             if waste_probability < STRONG_WASTE_THRESHOLD:
#                 return {
#                     "valid": False,
#                     "reason": (
#                         "Waste could not be identified "
#                         "with sufficient confidence."
#                     ),
#                     "wasteType": None,
#                     "severity": None,
#                     "confidence": int(
#                         round(waste_probability * 100)
#                     ),
#                     "engine": "ai",
#                     "details": [],
#                     "summary": (
#                         "The AI models were not confident "
#                         "enough to identify waste."
#                     ),
#                 }

#         # Severity based on number of detected objects.
#         if len(detections) >= 5:
#             severity = "High"
#         elif len(detections) >= 2:
#             severity = "Medium"
#         else:
#             severity = "Low"

#         details = [
#             {
#                 "label": detection["type"],
#                 "count": 1,
#                 "conf": int(
#                     round(detection["confidence"] * 100)
#                 ),
#             }
#             for detection in detections
#         ]

#         summary = (
#             f"{len(detections)} waste item"
#             f"{'s' if len(detections) != 1 else ''} detected. "
#             f"Primary type: {waste_type}."
#         )

#         return {
#             "valid": True,
#             "reason": None,
#             "wasteType": waste_type,
#             "severity": severity,
#             "confidence": int(
#                 round(highest_confidence * 100)
#             ),
#             "engine": "ai",
#             "details": details,
#             "summary": summary,
#         }
#     # ---------------------------------------------------------
#     # STEP 4: NO YOLO OBJECT DETECTED
#     # ---------------------------------------------------------
#     #
#     # The scene classifier must NOT be allowed to classify a
#     # normal image as waste by itself.
#     #
#     # Waste must be confirmed by an actual YOLO waste detection.

#     if not detections:
#         return {
#             "valid": False,
#             "reason": "No waste object was detected in the image.",
#             "wasteType": None,
#             "severity": None,
#             "confidence": int(
#                 round(waste_probability * 100)
#             ),
#             "engine": "ai",
#             "details": [],
#             "summary": (
#                 "The image does not contain a sufficiently "
#                 "confident detectable waste object."
#             ),
#         }

#     # ---------------------------------------------------------
#     # STEP 5: STRONG WASTE SCENE
#     # ---------------------------------------------------------
#     scene, scene_confidence = analyze_scene(
#         scene_feature_extractor,
#         scene_classifier,
#         scene_classes,
#         image,
#     )

#     # Scene model must also be confident.
#     if scene_confidence < SCENE_CONFIDENCE:
#         return {
#             "valid": False,
#             "reason": (
#                 "Waste scene could not be identified "
#                 "with sufficient confidence."
#             ),
#             "wasteType": None,
#             "severity": None,
#             "confidence": int(
#                 round(scene_confidence * 100)
#             ),
#             "engine": "ai",
#             "details": [],
#             "summary": (
#                 "The AI could not confidently identify "
#                 "the waste scene."
#             ),
#         }

#     amount = estimate_scene_amount(scene)

#     amount_to_severity = {
#         "SMALL": "Low",
#         "MEDIUM": "Medium",
#         "LARGE": "High",
#         "VERY LARGE": "High",
#     }

#     severity = amount_to_severity.get(
#         amount,
#         "Medium",
#     )

#     vehicle = recommend_vehicle(amount)

#     return {
#         "valid": True,
#         "reason": None,
#         "wasteType": None,
#         "severity": severity,
#         "confidence": int(
#             round(scene_confidence * 100)
#         ),
#         "engine": "ai",
#         "details": [
#             {
#                 "label": scene,
#                 "count": 1,
#                 "conf": int(
#                     round(scene_confidence * 100)
#                 ),
#             }
#         ],
#         "summary": (
#             f"Scene classified as {scene}. "
#             f"Estimated waste amount: {amount}. "
#             f"Recommended vehicle: "
#             f"{vehicle['vehicle']}."
#         ),
#     }
# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage:")
#         print(
#             'python ai\\src\\analyze_image.py '
#             '"path\\to\\image.jpg"'
#         )
#         sys.exit(1)

#     analyze_image(sys.argv[1])

import io
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import mobilenet_v3_small
from PIL import Image
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parents[1]

WASTE_GATE_MODEL = BASE_DIR / "models" / "waste_gate_v2_best.pth"

YOLO_MODEL = (
    BASE_DIR 
    / "models"
    / "waste_types_best.pt"
)

SCENE_MODEL = BASE_DIR / "models" / "scene_analysis_resnet18_best.pth"

WASTE_THRESHOLD = 0.40
STRONG_WASTE_THRESHOLD = 0.65
YOLO_CONFIDENCE = 0.50
YOLO_STRONG_THRESHOLD = 0.60
SCENE_CONFIDENCE = 0.70

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SCENE_CLASSES = [
    "construction_debris",
    "drain_blockage",
    "garbage_dump",
    "overflowing_bin",
]

scene_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

waste_gate_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_waste_gate():
    print("Loading Waste Gate V2...")

    model = mobilenet_v3_small(
        weights=None,
        num_classes=2,
    )

    checkpoint = torch.load(
        WASTE_GATE_MODEL,
        map_location=DEVICE,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


def load_yolo():
    print("Loading YOLO waste detector...")
    return YOLO(str(YOLO_MODEL))


def load_scene_model():
    print("Loading scene analysis model...")

    checkpoint = torch.load(
        SCENE_MODEL,
        map_location=DEVICE,
    )

    classes = checkpoint.get("classes", SCENE_CLASSES)

    resnet = models.resnet18(weights=None)

    feature_extractor = nn.Sequential(
        *list(resnet.children())[:-1]
    )
    feature_extractor.to(DEVICE)
    feature_extractor.eval()

    classifier = nn.Linear(
        resnet.fc.in_features,
        len(classes),
    )

    classifier.load_state_dict(
        checkpoint["classifier_state_dict"]
    )
    classifier.to(DEVICE)
    classifier.eval()

    return feature_extractor, classifier, classes


def check_waste(model, image):
    image_tensor = (
        waste_gate_transform(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    non_waste_probability = probabilities[0].item()
    waste_probability = probabilities[1].item()

    return (
        waste_probability >= WASTE_THRESHOLD,
        waste_probability,
        non_waste_probability,
    )


# def analyze_waste(yolo_model, image_path):
    # results = yolo_model.predict(
        # source=str(image_path),
        # conf=YOLO_CONFIDENCE,
        # verbose=False,
    # )
# 
    # result = results[0]
    # detections = []
# 
    # if result.boxes is None:
        # return detections, result.orig_shape
# 
    # for box in result.boxes:
        # class_id = int(box.cls[0].item())
        # confidence = float(box.conf[0].item())
        # xyxy = box.xyxy[0].tolist()
# 
        # detections.append({
            # "type": result.names[class_id],
            # "confidence": confidence,
            # "box": xyxy,
        # })
# 
    # return detections, result.orig_shape

def analyze_waste(yolo_model, image):
    results = yolo_model.predict(
        source=image,
        conf=YOLO_CONFIDENCE,
        verbose=False,
    )

    result = results[0]
    detections = []

    if result.boxes is None:
        return detections, result.orig_shape

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        xyxy = box.xyxy[0].tolist()

        detections.append({
            "type": result.names[class_id],
            "confidence": confidence,
            "box": xyxy,
        })

    return detections, result.orig_shape

def analyze_scene(feature_extractor, classifier, classes, image):
    image_tensor = (
        scene_transform(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():
        features = feature_extractor(image_tensor)
        features = features.view(features.size(0), -1)
        outputs = classifier(features)
        probabilities = torch.softmax(outputs, dim=1)[0]

        predicted_index = int(
            torch.argmax(probabilities).item()
        )

    scene = classes[predicted_index]
    confidence = probabilities[predicted_index].item()

    return scene, confidence


def estimate_scene_amount(scene):
    scene_amounts = {
        "garbage_dump": "VERY LARGE",
        "overflowing_bin": "LARGE",
        "construction_debris": "LARGE",
        "drain_blockage": "MEDIUM",
    }

    return scene_amounts.get(scene, "MEDIUM")


def recommend_vehicle(amount):
    recommendations = {
        "SMALL": {
            "vehicle": "Small waste collection vehicle",
            "reason": "Low visible waste accumulation.",
        },
        "MEDIUM": {
            "vehicle": "Medium waste collection vehicle",
            "reason": "Moderate visible waste accumulation.",
        },
        "LARGE": {
            "vehicle": "Large waste collection vehicle",
            "reason": "Large visible waste accumulation.",
        },
        "VERY LARGE": {
            "vehicle": "Large-capacity waste truck",
            "reason": (
                "Very large waste accumulation requires "
                "a high-capacity vehicle."
            ),
        },
    }

    return recommendations.get(
        amount,
        {
            "vehicle": "Vehicle recommendation unavailable",
            "reason": "Unknown waste amount category.",
        },
    )


def calculate_union_area(boxes, image_width, image_height):
    if not boxes:
        return 0.0

    rectangles = []

    for box in boxes:
        x1, y1, x2, y2 = box

        x1 = max(0, min(x1, image_width))
        y1 = max(0, min(y1, image_height))
        x2 = max(0, min(x2, image_width))
        y2 = max(0, min(y2, image_height))

        if x2 > x1 and y2 > y1:
            rectangles.append((x1, y1, x2, y2))

    if not rectangles:
        return 0.0

    x_values = sorted({
        value
        for x1, _, x2, _ in rectangles
        for value in (x1, x2)
    })

    total_area = 0.0

    for i in range(len(x_values) - 1):
        left = x_values[i]
        right = x_values[i + 1]
        width = right - left

        if width <= 0:
            continue

        active_intervals = [
            (y1, y2)
            for x1, y1, x2, y2 in rectangles
            if x1 < right and x2 > left
        ]

        if not active_intervals:
            continue

        active_intervals.sort()

        current_start, current_end = active_intervals[0]
        covered_height = 0.0

        for start, end in active_intervals[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                covered_height += current_end - current_start
                current_start = start
                current_end = end

        covered_height += current_end - current_start
        total_area += width * covered_height

    return total_area


def calculate_waste_coverage(detections, image_width, image_height):
    boxes = [detection["box"] for detection in detections]

    waste_area = calculate_union_area(
        boxes,
        image_width,
        image_height,
    )

    total_image_area = image_width * image_height

    if total_image_area <= 0:
        return 0.0

    return (waste_area / total_image_area) * 100


def estimate_waste_amount(coverage, detections):
    object_count = len(detections)

    if object_count == 1:
        return "SMALL"

    if object_count <= 3:
        return "SMALL" if coverage < 25 else "MEDIUM"

    if object_count <= 7:
        return "MEDIUM" if coverage < 20 else "LARGE"

    if object_count <= 15:
        return "LARGE" if coverage < 20 else "VERY LARGE"

    return "VERY LARGE"


def _non_waste_result(
    waste_probability,
    non_waste_probability,
    reason,
    summary,
):
    """Return a consistent NON-WASTE response."""
    return {
        "valid": False,
        "reason": reason,
        "wasteType": None,
        "severity": None,
        "confidence": int(
            round(non_waste_probability * 100)
        ),
        "engine": "ai",
        "details": [],
        "summary": summary,
    }


def _waste_result(
    detections,
    waste_probability,
):
    """Build the final WASTE response from confirmed YOLO detections."""
    type_mapping = {
        "hazardous-waste": "Hazardous",
        "medical-waste": "Hazardous",
        "organic-waste": "Organic",

        "recyclable-waste-cardboard": "Plastic",
        "recyclable-waste-clothes": "Plastic",
        "recyclable-waste-glass": "Plastic",
        "recyclable-waste-metal": "Plastic",
        "recyclable-waste-nylonbag": "Plastic",
        "recyclable-waste-paper": "Plastic",
        "recyclable-waste-paperbag": "Plastic",
        "recyclable-waste-plastic": "Plastic",
        "recyclable-waste-shoe": "Plastic",
    }

    type_counts = {}

    for detection in detections:
        model_type = detection["type"]

        app_type = type_mapping.get(
            model_type,
            "Plastic",
        )

        type_counts[app_type] = (
            type_counts.get(app_type, 0) + 1
        )

    waste_type = max(
        type_counts,
        key=type_counts.get,
    )

    highest_confidence = max(
        detection["confidence"]
        for detection in detections
    )

    if len(detections) >= 5:
        severity = "High"
    elif len(detections) >= 2:
        severity = "Medium"
    else:
        severity = "Low"

    details = [
        {
            "label": detection["type"],
            "count": 1,
            "conf": int(
                round(detection["confidence"] * 100)
            ),
        }
        for detection in detections
    ]

    summary = (
        f"{len(detections)} waste item"
        f"{'s' if len(detections) != 1 else ''} detected. "
        f"Primary type: {waste_type}."
    )

    return {
        "valid": True,
        "reason": None,
        "wasteType": waste_type,
        "severity": severity,
        "confidence": int(
            round(highest_confidence * 100)
        ),
        "engine": "ai",
        "details": details,
        "summary": summary,
    }


def analyze_image(image_path):
    """
    Command-line image analysis.

    Decision rule:
        1. Waste Gate filters obvious non-waste images.
        2. YOLO must confirm at least one waste object.
        3. Scene classification is NOT allowed to turn an image
           with zero YOLO detections into WASTE.

    This prevents normal camera images from being classified as
    waste simply because the scene model is forced to choose one
    of its waste-scene classes.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return

    print("\n" + "=" * 60)
    print("SWACHHLENS IMAGE ANALYSIS")
    print("=" * 60)
    print(f"Image: {image_path.name}")
    print(f"Device: {DEVICE}")

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as error:
        print(f"ERROR: Could not open image: {error}")
        return

    waste_gate = load_waste_gate()
    yolo_model = load_yolo()

    print("\n" + "-" * 60)
    print("STEP 1: WASTE GATE")
    print("-" * 60)

    is_waste, waste_probability, non_waste_probability = check_waste(
        waste_gate,
        image,
    )

    print(f"Waste probability: {waste_probability * 100:.2f}%")
    print(
        f"Non-waste probability: "
        f"{non_waste_probability * 100:.2f}%"
    )

    if not is_waste:
        print("\nResult: NON-WASTE")
        print("Please upload an image containing visible waste.")
        print("=" * 60)
        return

    print("\nWaste Gate: POSSIBLE WASTE")
    print("Proceeding to object confirmation...")

    print("\n" + "-" * 60)
    print("STEP 2: WASTE OBJECT CONFIRMATION")
    print("-" * 60)

    detections, image_shape = analyze_waste(
        yolo_model,
        image,
    )

    if not detections:
        print("\nResult: NON-WASTE")
        print(
            "No waste object was detected by YOLO "
            f"at the {YOLO_CONFIDENCE * 100:.0f}% threshold."
        )
        print(
            "The scene model was not used to override "
            "this decision."
        )
        print("=" * 60)
        return

    print(f"\nWaste objects detected: {len(detections)}")

    for index, detection in enumerate(
        detections,
        start=1,
    ):
        print(
            f"{index}. {detection['type']} "
            f"({detection['confidence'] * 100:.2f}%)"
        )

    highest_confidence = max(
        detection["confidence"]
        for detection in detections
    )

    if highest_confidence < YOLO_CONFIDENCE:
        print("\nResult: NON-WASTE")
        print(
            "A possible object was detected, but its "
            "confidence was too low to confirm waste."
        )
        print("=" * 60)
        return

    image_height, image_width = image_shape

    coverage = calculate_waste_coverage(
        detections,
        image_width,
        image_height,
    )

    amount = estimate_waste_amount(
        coverage,
        detections,
    )

    vehicle = recommend_vehicle(amount)

    print("\n" + "-" * 60)
    print("STEP 3: VISIBLE WASTE AMOUNT ESTIMATION")
    print("-" * 60)
    print(f"Image dimensions: {image_width} x {image_height}")
    print(f"Visible waste coverage: {coverage:.2f}%")
    print(f"Estimated amount: {amount}")

    print("\n" + "-" * 60)
    print("STEP 4: VEHICLE RECOMMENDATION")
    print("-" * 60)
    print(f"Recommended vehicle: {vehicle['vehicle']}")
    print(f"Reason: {vehicle['reason']}")

    print("\n" + "=" * 60)
    print("SWACHHLENS FINAL RESULT")
    print("=" * 60)
    print("Waste detected: YES")
    print(
        f"Waste confidence: "
        f"{waste_probability * 100:.2f}%"
    )
    print(f"Waste objects detected: {len(detections)}")
    print(f"Highest YOLO confidence: {highest_confidence * 100:.2f}%")
    print(f"Visible waste coverage: {coverage:.2f}%")
    print(f"Estimated amount: {amount}")
    print(f"Recommended vehicle: {vehicle['vehicle']}")
    print("=" * 60)


def analyze_image_bytes(image_bytes):
    """
    Backend-friendly entry point.

    Decision rule:
        1. Waste Gate must consider the image potentially waste.
        2. YOLO must confirm at least one waste object.
        3. If YOLO finds nothing, the image is NON-WASTE.
        4. Scene analysis is never used as a binary waste override.

    This prevents ordinary camera images from being accepted as
    waste merely because the scene classifier predicts a waste scene.
    """
    try:
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")
    except Exception as error:
        return {
            "valid": False,
            "reason": "The uploaded image could not be opened.",
            "wasteType": None,
            "severity": None,
            "confidence": 0,
            "engine": "ai",
            "details": [],
            "summary": f"Invalid image: {error}",
        }

    waste_gate = load_waste_gate()
    yolo_model = load_yolo()

    # ---------------------------------------------------------
    # STEP 1: WASTE GATE
    # ---------------------------------------------------------
    is_waste, waste_probability, non_waste_probability = check_waste(
        waste_gate,
        image,
    )

    if waste_probability < WASTE_THRESHOLD:
        return _non_waste_result(
            waste_probability,
            non_waste_probability,
            "Image does not appear to contain waste.",
            "Image rejected by the waste gate.",
        )

    # ---------------------------------------------------------
    # STEP 2: YOLO WASTE DETECTION
    # ---------------------------------------------------------
    detections, original_shape = analyze_waste(
        yolo_model,
        image,
    )

    # ---------------------------------------------------------
    # STEP 3: ACTUAL WASTE OBJECT CONFIRMATION
    # ---------------------------------------------------------
    #
    # This is the important fix:
    # a high Waste Gate score alone is NOT enough.
    # If YOLO detects no waste object, the image is NON-WASTE.
    #
    if not detections:
        return _non_waste_result(
            waste_probability,
            non_waste_probability,
            "No waste object was detected in the image.",
            (
                "The Waste Gate detected possible waste, "
                "but YOLO could not confirm a waste object."
            ),
        )

    highest_confidence = max(
        detection["confidence"]
        for detection in detections
    )

    if highest_confidence < YOLO_CONFIDENCE:
        return _non_waste_result(
            waste_probability,
            non_waste_probability,
            "Waste object confidence is too low.",
            (
                "A possible object was detected, "
                "but confidence was too low to confirm waste."
            ),
        )

    return _waste_result(
        detections,
        waste_probability,
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:")
        print(
            'python ai\\src\\analyze_image.py '
            '"path\\to\\image.jpg"'
        )
        sys.exit(1)

    analyze_image(sys.argv[1])