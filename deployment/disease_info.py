"""
disease_info.py
================
Local knowledge base for LeafFusionNet.

This module exposes a single dictionary, ``DISEASE_INFO``, that maps every
PlantVillage class label used by the LeafFusionNet classifier to structured,
human-readable agricultural information (display name, crop, severity,
description, treatment steps, and prevention steps).

This module is read-only reference data. It has no side effects, no external
dependencies, and does not import anything from the rest of the LeafFusionNet
codebase, so it can be safely imported from anywhere (e.g. predictor.py,
API route handlers, reporting utilities) without risk of circular imports.
"""

from typing import Dict, List, TypedDict


class DiseaseInfoEntry(TypedDict):
    """Structured information describing a single PlantVillage class."""

    display_name: str
    crop: str
    severity: str  # One of: "Low", "Medium", "High", "Healthy"
    description: str
    treatment: List[str]
    prevention: List[str]


DISEASE_INFO: Dict[str, DiseaseInfoEntry] = {
    # ------------------------------------------------------------------ #
    # Apple
    # ------------------------------------------------------------------ #
    "Apple___Apple_scab": {
        "display_name": "Apple Scab",
        "crop": "Apple",
        "severity": "Medium",
        "description": (
            "A fungal disease caused by Venturia inaequalis that produces "
            "olive-green to dark scabby lesions on leaves and fruit, "
            "leading to premature leaf drop and blemished fruit."
        ),
        "treatment": [
            "Apply a labeled fungicide (e.g. captan or myclobutanil) at "
            "green tip and continue on a 7-10 day schedule during wet periods.",
            "Remove and destroy infected leaves and fruit from the tree.",
            "Prune out heavily infected shoots to reduce inoculum.",
        ],
        "prevention": [
            "Rake up and dispose of fallen leaves each autumn to reduce "
            "overwintering spores.",
            "Choose scab-resistant apple cultivars where possible.",
            "Ensure good air circulation through proper pruning and spacing.",
        ],
    },
    "Apple___Black_rot": {
        "display_name": "Apple Black Rot",
        "crop": "Apple",
        "severity": "High",
        "description": (
            "A fungal disease caused by Botryosphaeria obtusa that causes "
            "purple leaf spots, fruit rot with concentric rings, and "
            "cankers on branches."
        ),
        "treatment": [
            "Prune out and destroy cankered wood and mummified fruit.",
            "Apply fungicides labeled for black rot starting at petal fall.",
            "Remove infected fruit promptly to limit spore spread.",
        ],
        "prevention": [
            "Sanitize the orchard by removing dead wood and mummified fruit "
            "every dormant season.",
            "Avoid tree stress through adequate watering and fertilization.",
            "Maintain good pruning to improve airflow and sun exposure.",
        ],
    },
    "Apple___Cedar_apple_rust": {
        "display_name": "Apple Cedar Rust",
        "crop": "Apple",
        "severity": "Medium",
        "description": (
            "A fungal disease caused by Gymnosporangium juniperi-virginianae "
            "that alternates between apple and cedar/juniper hosts, causing "
            "bright orange-yellow leaf spots on apple."
        ),
        "treatment": [
            "Apply a protective fungicide from pink bud stage through "
            "several weeks after petal fall.",
            "Remove nearby galls from cedar/juniper trees when feasible.",
            "Remove severely infected leaves to reduce disease pressure.",
        ],
        "prevention": [
            "Plant rust-resistant apple varieties where available.",
            "Avoid planting apple trees near cedar or juniper trees "
            "(ideally more than a few hundred meters away).",
            "Monitor trees closely during wet spring weather.",
        ],
    },
    "Apple___healthy": {
        "display_name": "Apple - Healthy",
        "crop": "Apple",
        "severity": "Healthy",
        "description": "This leaf appears healthy.",
        "treatment": ["No treatment required."],
        "prevention": ["Continue regular crop monitoring."],
    },
    # ------------------------------------------------------------------ #
    # Cherry
    # ------------------------------------------------------------------ #
    "Cherry_(including_sour)___Powdery_mildew": {
        "display_name": "Cherry Powdery Mildew",
        "crop": "Cherry",
        "severity": "Medium",
        "description": (
            "A fungal disease caused by Podosphaera clandestina producing "
            "white powdery patches on leaves and shoots, which can reduce "
            "fruit quality and vigor."
        ),
        "treatment": [
            "Apply sulfur-based or other labeled fungicides at first sign "
            "of infection.",
            "Prune out and remove heavily infected shoots.",
            "Improve air circulation by thinning dense canopy growth.",
        ],
        "prevention": [
            "Select resistant cherry varieties when planting.",
            "Avoid excessive nitrogen fertilization that promotes soft "
            "susceptible growth.",
            "Space and prune trees for good airflow and sunlight penetration.",
        ],
    },
    "Cherry_(including_sour)___healthy": {
        "display_name": "Cherry - Healthy",
        "crop": "Cherry",
        "severity": "Healthy",
        "description": "This leaf appears healthy.",
        "treatment": ["No treatment required."],
        "prevention": ["Continue regular crop monitoring."],
    },
    # ------------------------------------------------------------------ #
    # Corn (Maize)
    # ------------------------------------------------------------------ #
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "display_name": "Corn Gray Leaf Spot",
        "crop": "Corn",
        "severity": "Medium",
        "description": (
            "A fungal disease caused by Cercospora zeae-maydis that "
            "produces rectangular tan-to-gray lesions running parallel to "
            "leaf veins, reducing photosynthetic area."
        ),
        "treatment": [
            "Apply a foliar fungicide when lesions appear on upper leaves "
            "before tasseling.",
            "Rotate to a non-host crop for at least one season in "
            "heavily infected fields.",
            "Remove or bury infected crop residue after harvest.",
        ],
        "prevention": [
            "Plant resistant or tolerant corn hybrids.",
            "Practice crop rotation with non-host crops.",
            "Use tillage to reduce residue-borne inoculum where appropriate.",
        ],
    },
    "Corn_(maize)___Common_rust_": {
        "display_name": "Corn Common Rust",
        "crop": "Corn",
        "severity": "Low",
        "description": (
            "A fungal disease caused by Puccinia sorghi that produces "
            "small, reddish-brown, powdery pustules scattered on both "
            "leaf surfaces."
        ),
        "treatment": [
            "Apply a foliar fungicide if pustules are numerous on young "
            "plants before tasseling.",
            "Monitor fields regularly during humid, cool weather.",
            "Remove severely rusted leaves in small-scale plantings.",
        ],
        "prevention": [
            "Plant rust-resistant corn hybrids.",
            "Avoid late planting that exposes young plants to peak rust "
            "spore periods.",
            "Maintain balanced fertilization to support plant vigor.",
        ],
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "display_name": "Corn Northern Leaf Blight",
        "crop": "Corn",
        "severity": "High",
        "description": (
            "A fungal disease caused by Exserohilum turcicum that causes "
            "long, cigar-shaped gray-green to tan lesions on leaves, "
            "which can significantly reduce yield."
        ),
        "treatment": [
            "Apply a foliar fungicide at early disease onset, especially "
            "on susceptible hybrids.",
            "Rotate crops away from corn for one to two seasons.",
            "Till under infected residue to speed decomposition.",
        ],
        "prevention": [
            "Plant hybrids with genetic resistance to Northern Leaf Blight.",
            "Practice crop rotation and residue management.",
            "Scout fields regularly during the growing season.",
        ],
    },
    "Corn_(maize)___healthy": {
        "display_name": "Corn - Healthy",
        "crop": "Corn",
        "severity": "Healthy",
        "description": "This leaf appears healthy.",
        "treatment": ["No treatment required."],
        "prevention": ["Continue regular crop monitoring."],
    },
    # ------------------------------------------------------------------ #
    # Grape
    # ------------------------------------------------------------------ #
    "Grape___Black_rot": {
        "display_name": "Grape Black Rot",
        "crop": "Grape",
        "severity": "High",
        "description": (
            "A fungal disease caused by Guignardia bidwellii that causes "
            "circular tan leaf spots with dark borders and shrivels fruit "
            "into hard black mummies."
        ),
        "treatment": [
            "Apply fungicides labeled for black rot starting at bud break "
            "and continuing through fruit set.",
            "Remove and destroy mummified berries and infected leaves.",
            "Prune out infected canes during dormancy.",
        ],
        "prevention": [
            "Maintain an open canopy through proper pruning for airflow.",
            "Remove mummified fruit and leaf litter each season.",
            "Choose resistant grape varieties where available.",
        ],
    },
    "Grape___Esca_(Black_Measles)": {
        "display_name": "Grape Esca (Black Measles)",
        "crop": "Grape",
        "severity": "High",
        "description": (
            "A fungal trunk disease complex causing tiger-stripe leaf "
            "discoloration, dark spotting on berries, and can lead to "
            "sudden vine collapse."
        ),
        "treatment": [
            "Prune out and destroy visibly diseased or dead wood.",
            "Apply trunk-protectant fungicide pastes to large pruning "
            "wounds.",
            "Remove and destroy severely affected vines to limit spread.",
        ],
        "prevention": [
            "Prune during dry weather to reduce infection risk through "
            "wounds.",
            "Avoid large pruning cuts on mature trunks when possible.",
            "Use certified disease-free planting material.",
        ],
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "display_name": "Grape Leaf Blight (Isariopsis Leaf Spot)",
        "crop": "Grape",
        "severity": "Medium",
        "description": (
            "A fungal disease caused by Pseudocercospora vitis that "
            "produces angular reddish-brown leaf spots, leading to "
            "premature defoliation in severe cases."
        ),
        "treatment": [
            "Apply a labeled fungicide during periods of high humidity "
            "or rainfall.",
            "Remove and destroy heavily infected leaves.",
            "Improve canopy airflow through leaf pulling and pruning.",
        ],
        "prevention": [
            "Practice good sanitation by removing fallen leaves each season.",
            "Avoid overhead irrigation that keeps foliage wet.",
            "Space and trellis vines to maximize air circulation.",
        ],
    },
    "Grape___healthy": {
        "display_name": "Grape - Healthy",
        "crop": "Grape",
        "severity": "Healthy",
        "description": "This leaf appears healthy.",
        "treatment": ["No treatment required."],
        "prevention": ["Continue regular crop monitoring."],
    },
    # ------------------------------------------------------------------ #
    # Peach
    # ------------------------------------------------------------------ #
    "Peach___Bacterial_spot": {
        "display_name": "Peach Bacterial Spot",
        "crop": "Peach",
        "severity": "Medium",
        "description": (
            "A bacterial disease caused by Xanthomonas arboricola pv. "
            "pruni that produces small, angular, water-soaked leaf spots "
            "and pitted lesions on fruit."
        ),
        "treatment": [
            "Apply copper-based bactericides during dormant and early "
            "growing season, following label guidance to avoid phytotoxicity.",
            "Remove and destroy severely infected leaves and fruit.",
            "Avoid working in the orchard when foliage is wet.",
        ],
        "prevention": [
            "Plant bacterial-spot-resistant peach varieties.",
            "Avoid excessive nitrogen fertilization.",
            "Use windbreaks and proper spacing to reduce leaf wetness "
            "duration.",
        ],
    },
    "Peach___healthy": {
        "display_name": "Peach - Healthy",
        "crop": "Peach",
        "severity": "Healthy",
        "description": "This leaf appears healthy.",
        "treatment": ["No treatment required."],
        "prevention": ["Continue regular crop monitoring."],
    },
    # ------------------------------------------------------------------ #
    # Pepper (Bell)
    # ------------------------------------------------------------------ #
    "Pepper,_bell___Bacterial_spot": {
        "display_name": "Bell Pepper Bacterial Spot",
        "crop": "Pepper (Bell)",
        "severity": "Medium",
        "description": (
            "A bacterial disease caused by Xanthomonas species that "
            "produces small, water-soaked, dark lesions on leaves and "
            "fruit, which can cause defoliation in severe cases."
        ),
        "treatment": [
            "Apply copper-based bactericides at early symptom onset, "
            "combined with a mancozeb tank-mix where permitted.",
            "Remove and destroy heavily infected plants.",
            "Avoid overhead irrigation to reduce bacterial spread.",
        ],
        "prevention": [
            "Use certified disease-free seed and transplants.",
            "Practice crop rotation with non-host crops for at least "
            "one to two years.",
            "Avoid working with plants when foliage is wet.",
        ],
    },
    "Pepper,_bell___healthy": {
        "display_name": "Bell Pepper - Healthy",
        "crop": "Pepper (Bell)",
        "severity": "Healthy",
        "description": "This leaf appears healthy.",
        "treatment": ["No treatment required."],
        "prevention": ["Continue regular crop monitoring."],
    },
    # ------------------------------------------------------------------ #
    # Potato
    # ------------------------------------------------------------------ #
    "Potato___Early_blight": {
        "display_name": "Potato Early Blight",
        "crop": "Potato",
        "severity": "Medium",
        "description": (
            "A fungal disease caused by Alternaria solani that produces "
            "dark, concentric-ringed 'target' spots on lower, older "
            "leaves first."
        ),
        "treatment": [
            "Apply a labeled fungicide (e.g. chlorothalonil or "
            "azoxystrobin) at first sign of symptoms.",
            "Remove and destroy severely infected lower leaves.",
            "Ensure adequate plant nutrition, especially nitrogen, to "
            "reduce susceptibility.",
        ],
        "prevention": [
            "Rotate potatoes with non-solanaceous crops for at least "
            "two years.",
            "Use certified disease-free seed potatoes.",
            "Avoid overhead irrigation and water in the morning to allow "
            "foliage to dry.",
        ],
    },
    "Potato___Late_blight": {
        "display_name": "Potato Late Blight",
        "crop": "Potato",
        "severity": "High",
        "description": (
            "A destructive disease caused by the oomycete Phytophthora "
            "infestans that produces dark, water-soaked lesions on "
            "leaves and can rapidly destroy entire fields."
        ),
        "treatment": [
            "Apply a systemic or protectant fungicide immediately upon "
            "detection and continue on a strict schedule.",
            "Remove and destroy infected plants or entire affected "
            "sections of the field.",
            "Destroy volunteer potato plants and cull piles.",
        ],
        "prevention": [
            "Plant certified, disease-free seed potatoes.",
            "Use resistant potato varieties where available.",
            "Monitor weather-based blight forecasts and apply preventive "
            "fungicides ahead of favorable disease conditions.",
        ],
    },
    "Potato___healthy": {
        "display_name": "Potato - Healthy",
        "crop": "Potato",
        "severity": "Healthy",
        "description": "This leaf appears healthy.",
        "treatment": ["No treatment required."],
        "prevention": ["Continue regular crop monitoring."],
    },
    # ------------------------------------------------------------------ #
    # Strawberry
    # ------------------------------------------------------------------ #
    "Strawberry___Leaf_scorch": {
        "display_name": "Strawberry Leaf Scorch",
        "crop": "Strawberry",
        "severity": "Medium",
        "description": (
            "A fungal disease caused by Diplocarpon earlianum that "
            "produces small purple spots that merge, giving leaves a "
            "scorched appearance."
        ),
        "treatment": [
            "Apply a labeled fungicide at early symptom onset and "
            "repeat per label interval.",
            "Remove and destroy infected leaves after harvest (renovation).",
            "Avoid overhead irrigation late in the day.",
        ],
        "prevention": [
            "Choose resistant strawberry cultivars where available.",
            "Space plants adequately to promote airflow.",
            "Remove old and infected foliage during annual renovation.",
        ],
    },
    "Strawberry___healthy": {
        "display_name": "Strawberry - Healthy",
        "crop": "Strawberry",
        "severity": "Healthy",
        "description": "This leaf appears healthy.",
        "treatment": ["No treatment required."],
        "prevention": ["Continue regular crop monitoring."],
    },
    # ------------------------------------------------------------------ #
    # Tomato
    # ------------------------------------------------------------------ #
    "Tomato___Bacterial_spot": {
        "display_name": "Tomato Bacterial Spot",
        "crop": "Tomato",
        "severity": "Medium",
        "description": (
            "A bacterial disease caused by Xanthomonas species producing "
            "small, dark, water-soaked spots on leaves, stems, and fruit."
        ),
        "treatment": [
            "Apply copper-based bactericides at first symptom onset.",
            "Remove and destroy severely infected plant material.",
            "Avoid handling plants when leaves are wet.",
        ],
        "prevention": [
            "Use certified disease-free seed and transplants.",
            "Rotate crops with non-solanaceous plants for at least a year.",
            "Avoid overhead irrigation; use drip irrigation instead.",
        ],
    },
    "Tomato___Early_blight": {
        "display_name": "Tomato Early Blight",
        "crop": "Tomato",
        "severity": "Medium",
        "description": (
            "A fungal disease caused by Alternaria solani that produces "
            "dark, concentric 'target' spots on older leaves, often "
            "leading to defoliation."
        ),
        "treatment": [
            "Apply a labeled fungicide (e.g. chlorothalonil or "
            "mancozeb) at first symptom onset.",
            "Remove and destroy infected lower leaves promptly.",
            "Stake or cage plants to improve airflow and reduce soil "
            "splash.",
        ],
        "prevention": [
            "Rotate tomatoes with non-host crops for two to three years.",
            "Mulch around plants to reduce soil splash onto leaves.",
            "Avoid overhead irrigation; water at the base of plants.",
        ],
    },
    "Tomato___Late_blight": {
        "display_name": "Tomato Late Blight",
        "crop": "Tomato",
        "severity": "High",
        "description": (
            "A destructive disease caused by the oomycete Phytophthora "
            "infestans that produces large, dark, water-soaked lesions "
            "and can destroy plants within days under favorable conditions."
        ),
        "treatment": [
            "Apply a systemic or protectant fungicide immediately and "
            "continue on a strict preventive schedule.",
            "Remove and destroy infected plants promptly to limit spread.",
            "Avoid working in fields during wet conditions to prevent "
            "spreading spores.",
        ],
        "prevention": [
            "Plant resistant tomato varieties where available.",
            "Ensure good field drainage and airflow.",
            "Monitor regional blight forecasts and apply preventive "
            "sprays proactively.",
        ],
    },
    "Tomato___Leaf_Mold": {
        "display_name": "Tomato Leaf Mold",
        "crop": "Tomato",
        "severity": "Medium",
        "description": (
            "A fungal disease caused by Passalora fulva (syn. Fulvia "
            "fulva) that produces pale green-to-yellow spots on upper "
            "leaf surfaces with olive-green mold underneath, favored by "
            "high humidity."
        ),
        "treatment": [
            "Apply a labeled fungicide at first symptom onset, "
            "especially in greenhouses.",
            "Improve ventilation to reduce humidity around plants.",
            "Remove and destroy infected leaves.",
        ],
        "prevention": [
            "Use resistant tomato varieties where available.",
            "Increase plant spacing and prune for better airflow.",
            "Avoid overhead watering and reduce greenhouse humidity.",
        ],
    },
    "Tomato___Septoria_leaf_spot": {
        "display_name": "Tomato Septoria Leaf Spot",
        "crop": "Tomato",
        "severity": "Medium",
        "description": (
            "A fungal disease caused by Septoria lycopersici that "
            "produces small, circular spots with dark borders and gray "
            "centers, typically starting on lower leaves."
        ),
        "treatment": [
            "Apply a labeled fungicide such as chlorothalonil at first "
            "sign of spotting.",
            "Remove and destroy infected lower leaves.",
            "Avoid overhead watering to reduce leaf wetness.",
        ],
        "prevention": [
            "Rotate tomatoes with non-host crops for at least one year.",
            "Mulch to prevent soil-borne spores from splashing onto leaves.",
            "Stake plants and space adequately for good airflow.",
        ],
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "display_name": "Tomato Two-Spotted Spider Mite Damage",
        "crop": "Tomato",
        "severity": "Medium",
        "description": (
            "Feeding damage from Tetranychus urticae (two-spotted spider "
            "mite) causing fine stippling, yellowing, and fine webbing "
            "on leaves, especially in hot, dry conditions."
        ),
        "treatment": [
            "Apply a labeled miticide or insecticidal soap, targeting "
            "the undersides of leaves.",
            "Introduce predatory mites as a biological control option.",
            "Increase humidity around plants, as mites thrive in dry "
            "conditions.",
        ],
        "prevention": [
            "Monitor plants regularly, especially during hot, dry weather.",
            "Avoid excessive drought stress on plants.",
            "Remove heavily infested leaves early to slow population "
            "growth.",
        ],
    },
    "Tomato___Target_Spot": {
        "display_name": "Tomato Target Spot",
        "crop": "Tomato",
        "severity": "Medium",
        "description": (
            "A fungal disease caused by Corynespora cassiicola that "
            "produces brown lesions with concentric target-like rings "
            "on leaves, stems, and fruit."
        ),
        "treatment": [
            "Apply a labeled fungicide at early symptom onset.",
            "Remove and destroy infected plant debris.",
            "Improve air circulation through pruning and staking.",
        ],
        "prevention": [
            "Rotate crops with non-host plants.",
            "Avoid overhead irrigation and excess leaf wetness.",
            "Use resistant varieties where available.",
        ],
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "display_name": "Tomato Yellow Leaf Curl Virus",
        "crop": "Tomato",
        "severity": "High",
        "description": (
            "A viral disease transmitted by whiteflies (Bemisia tabaci) "
            "that causes upward leaf curling, yellowing, and severe "
            "stunting of plants."
        ),
        "treatment": [
            "Remove and destroy infected plants to reduce virus "
            "reservoirs; no chemical cure exists for infected plants.",
            "Control whitefly populations using insecticides or "
            "insecticidal soap.",
            "Use reflective mulches to deter whiteflies from settling.",
        ],
        "prevention": [
            "Plant virus-resistant tomato varieties where available.",
            "Use insect-proof netting or row covers on young plants.",
            "Manage whitefly populations proactively throughout the "
            "season.",
        ],
    },
    "Tomato___Tomato_mosaic_virus": {
        "display_name": "Tomato Mosaic Virus",
        "crop": "Tomato",
        "severity": "High",
        "description": (
            "A viral disease causing mottled light and dark green "
            "mosaic patterns on leaves, leaf distortion, and reduced "
            "fruit yield and quality."
        ),
        "treatment": [
            "Remove and destroy infected plants; no chemical cure exists "
            "for infected plants.",
            "Disinfect tools and hands after handling infected plants "
            "to prevent mechanical spread.",
            "Avoid using tobacco products near plants, as the virus can "
            "spread via tobacco.",
        ],
        "prevention": [
            "Use certified virus-free seed and resistant varieties.",
            "Practice strict sanitation of tools and hands between plants.",
            "Control weeds that may host the virus between growing "
            "seasons.",
        ],
    },
    "Tomato___healthy": {
        "display_name": "Tomato - Healthy",
        "crop": "Tomato",
        "severity": "Healthy",
        "description": "This leaf appears healthy.",
        "treatment": ["No treatment required."],
        "prevention": ["Continue regular crop monitoring."],
    },
}
