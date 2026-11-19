"""
utils.py — Helper Utilities for Plant Disease Detection System
CVRT Engineering College, Tadipatri | JNTUA B.Tech CSE 2024-25
"""

import os
import re
import json
from datetime import datetime

# ── Disease information database ──────────────────────────────────────────────
# Maps disease class names to human-readable info and treatment recommendations
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "display_name": "Apple Scab",
        "crop": "Apple",
        "description": "Apple scab is a fungal disease caused by Venturia inaequalis. It produces dark, scabby lesions on leaves and fruit.",
        "severity": "Moderate",
        "treatment": [
            "Apply fungicides containing captan or myclobutanil at bud break",
            "Remove and destroy infected leaves and fallen debris",
            "Prune trees to improve air circulation",
            "Apply copper-based sprays as preventive measure"
        ],
        "prevention": [
            "Plant scab-resistant apple varieties",
            "Ensure good drainage and air circulation",
            "Avoid overhead irrigation",
            "Monitor and spray during wet spring weather"
        ]
    },
    "Apple___Black_rot": {
        "display_name": "Apple Black Rot",
        "crop": "Apple",
        "description": "Black rot is caused by the fungus Botryosphaeria obtusa. Affects leaves, fruit, and bark with dark brown to black lesions.",
        "severity": "High",
        "treatment": [
            "Apply fungicides with thiophanate-methyl or captan",
            "Prune and remove all mummified fruit and infected wood",
            "Destroy infected plant material — do not compost",
            "Apply Bordeaux mixture as protective spray"
        ],
        "prevention": [
            "Remove mummified fruits from trees promptly",
            "Prune dead or diseased branches during dormancy",
            "Maintain proper tree nutrition",
            "Control other insects that cause wounds for fungal entry"
        ]
    },
    "Apple___Cedar_apple_rust": {
        "display_name": "Cedar Apple Rust",
        "crop": "Apple",
        "description": "Caused by the fungus Gymnosporangium juniperi-virginianae, requiring both apple and cedar/juniper trees to complete its life cycle.",
        "severity": "Moderate",
        "treatment": [
            "Apply fungicides (myclobutanil or propiconazole) from pink through third cover",
            "Remove nearby cedar or juniper trees if possible",
            "Use protective fungicide sprays during wet periods"
        ],
        "prevention": [
            "Plant rust-resistant apple varieties",
            "Remove cedar galls from nearby juniper trees in early spring",
            "Avoid planting apples near junipers/cedars"
        ]
    },
    "Apple___healthy": {
        "display_name": "Healthy Apple",
        "crop": "Apple",
        "description": "The apple leaf appears healthy with no visible signs of disease.",
        "severity": "None",
        "treatment": ["No treatment required"],
        "prevention": [
            "Continue regular monitoring",
            "Maintain good agricultural practices",
            "Ensure proper fertilization and irrigation"
        ]
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "display_name": "Corn Gray Leaf Spot",
        "crop": "Corn (Maize)",
        "description": "Gray leaf spot, caused by Cercospora zeae-maydis, produces rectangular gray-brown lesions that run parallel to leaf veins.",
        "severity": "High",
        "treatment": [
            "Apply foliar fungicides (strobilurins or triazoles) at VT/R1 growth stage",
            "Use fungicides containing azoxystrobin or propiconazole",
            "Apply when disease first appears on lower leaves"
        ],
        "prevention": [
            "Plant resistant hybrid varieties",
            "Practice crop rotation (non-corn crop for 1-2 years)",
            "Avoid no-till in high-risk fields",
            "Plant early to escape severe infection periods"
        ]
    },
    "Corn_(maize)___Common_rust_": {
        "display_name": "Corn Common Rust",
        "crop": "Corn (Maize)",
        "description": "Common rust is caused by Puccinia sorghi. Produces small, oval to elongated cinnamon-brown pustules on both leaf surfaces.",
        "severity": "Moderate",
        "treatment": [
            "Apply foliar fungicides if rust is found before tasseling",
            "Use strobilurin-based fungicides (azoxystrobin, pyraclostrobin)",
            "Treat early — fungicides are most effective before heavy infection"
        ],
        "prevention": [
            "Plant resistant hybrid varieties (most effective strategy)",
            "Early planting avoids peak rust season",
            "Monitor fields regularly from V6 onwards"
        ]
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "display_name": "Corn Northern Leaf Blight",
        "crop": "Corn (Maize)",
        "description": "Caused by Exserohilum turcicum, produces large, cigar-shaped gray-green or tan lesions on leaves.",
        "severity": "High",
        "treatment": [
            "Apply fungicides at tasseling if disease is detected",
            "Use fungicides with prothioconazole, azoxystrobin, or pyraclostrobin",
            "Focus spray on upper canopy where lesions spread most"
        ],
        "prevention": [
            "Plant resistant hybrids",
            "Rotate with non-corn crops",
            "Bury infected crop residue by tillage"
        ]
    },
    "Corn_(maize)___healthy": {
        "display_name": "Healthy Corn",
        "crop": "Corn (Maize)",
        "description": "The corn leaf appears healthy with no visible signs of disease.",
        "severity": "None",
        "treatment": ["No treatment required"],
        "prevention": ["Continue regular monitoring", "Maintain good field management practices"]
    },
    "Grape___Black_rot": {
        "display_name": "Grape Black Rot",
        "crop": "Grape",
        "description": "Black rot, caused by Guignardia bidwellii, is one of the most serious grape diseases. Produces dark brown lesions on leaves with tiny black pycnidia.",
        "severity": "High",
        "treatment": [
            "Apply fungicides containing mancozeb, captan, or myclobutanil",
            "Begin spray program at first sign of budbreak",
            "Remove and destroy infected mummified berries"
        ],
        "prevention": [
            "Prune vines to allow good air circulation",
            "Remove all mummified fruit from previous season",
            "Apply fungicides preventively during wet weather"
        ]
    },
    "Grape___Esca_(Black_Measles)": {
        "display_name": "Grape Esca (Black Measles)",
        "crop": "Grape",
        "description": "Esca is a complex disease caused by multiple wood-infecting fungi. Produces interveinal chlorosis forming a striped or mottled pattern.",
        "severity": "Severe",
        "treatment": [
            "No completely effective chemical cure exists",
            "Remove and destroy severely infected vines",
            "Apply sodium arsenite to pruning wounds (where legal)",
            "Injecting fungicides has shown limited success"
        ],
        "prevention": [
            "Protect pruning wounds with fungicidal wound protectants",
            "Avoid large pruning wounds and prune during dry weather",
            "Use clean certified propagation material"
        ]
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "display_name": "Grape Leaf Blight",
        "crop": "Grape",
        "description": "Caused by Pseudocercospora vitis, producing angular dark brown lesions on upper leaf surfaces, often surrounded by yellow halos.",
        "severity": "Moderate",
        "treatment": [
            "Apply copper-based fungicides or mancozeb",
            "Apply at first sign of disease and repeat every 14 days",
            "Remove heavily infected leaves"
        ],
        "prevention": [
            "Ensure proper vine training for air circulation",
            "Avoid wetting foliage during irrigation"
        ]
    },
    "Grape___healthy": {
        "display_name": "Healthy Grape",
        "crop": "Grape",
        "description": "The grape leaf appears healthy with no visible signs of disease.",
        "severity": "None",
        "treatment": ["No treatment required"],
        "prevention": ["Continue regular monitoring and standard vineyard management"]
    },
    "Potato___Early_blight": {
        "display_name": "Potato Early Blight",
        "crop": "Potato",
        "description": "Caused by Alternaria solani, early blight produces dark brown to black target-like lesions with concentric rings on lower leaves first.",
        "severity": "Moderate",
        "treatment": [
            "Apply fungicides (chlorothalonil, mancozeb, or azoxystrobin) at first signs",
            "Remove severely infected lower leaves",
            "Apply neem-based organic treatments as alternative"
        ],
        "prevention": [
            "Use certified disease-free seed potatoes",
            "Practice 2-3 year crop rotation",
            "Avoid overhead irrigation — use drip irrigation",
            "Ensure proper soil nutrition (adequate nitrogen and potassium)"
        ]
    },
    "Potato___Late_blight": {
        "display_name": "Potato Late Blight",
        "crop": "Potato",
        "description": "Caused by Phytophthora infestans — the organism responsible for the Irish Potato Famine. One of the most destructive agricultural diseases.",
        "severity": "Critical",
        "treatment": [
            "Apply fungicides immediately (metalaxyl, cymoxanil, or famoxadone+cymoxanil)",
            "Destroy infected plants — do NOT compost",
            "Apply copper-based fungicides as emergency treatment",
            "Harvest remaining healthy tubers immediately if infection is severe"
        ],
        "prevention": [
            "Plant certified blight-resistant varieties",
            "Avoid planting in poorly drained soils",
            "Monitor weather — high humidity + cool temperatures = high risk",
            "Apply preventive fungicides before disease onset in high-risk periods"
        ]
    },
    "Potato___healthy": {
        "display_name": "Healthy Potato",
        "crop": "Potato",
        "description": "The potato leaf appears healthy with no visible signs of disease.",
        "severity": "None",
        "treatment": ["No treatment required"],
        "prevention": ["Continue regular monitoring", "Maintain appropriate irrigation and nutrition"]
    },
    "Tomato___Bacterial_spot": {
        "display_name": "Tomato Bacterial Spot",
        "crop": "Tomato",
        "description": "Caused by Xanthomonas species, bacterial spot produces small, water-soaked, brown spots with yellow halos on leaves, stems, and fruit.",
        "severity": "Moderate to High",
        "treatment": [
            "Apply copper-based bactericides (copper hydroxide or copper sulfate)",
            "Combine with mancozeb for enhanced effectiveness",
            "Remove and destroy infected plant material"
        ],
        "prevention": [
            "Use certified disease-free seeds and transplants",
            "Avoid overhead irrigation",
            "Rotate crops annually",
            "Disinfect tools between plants"
        ]
    },
    "Tomato___Early_blight": {
        "display_name": "Tomato Early Blight",
        "crop": "Tomato",
        "description": "Early blight, caused by Alternaria solani, produces dark concentric rings forming target-like spots on older leaves.",
        "severity": "Moderate",
        "treatment": [
            "Apply fungicides (chlorothalonil or mancozeb) at first sign",
            "Apply copper-based spray as organic alternative",
            "Remove infected lower leaves"
        ],
        "prevention": [
            "Mulch around plant base to prevent soil splash",
            "Water at base, not overhead",
            "Stake or cage to keep foliage off the ground",
            "Plant resistant varieties"
        ]
    },
    "Tomato___Late_blight": {
        "display_name": "Tomato Late Blight",
        "crop": "Tomato",
        "description": "Caused by Phytophthora infestans, producing water-soaked gray-green or brown patches on leaves that quickly turn brown and die.",
        "severity": "Critical",
        "treatment": [
            "Apply fungicides immediately (chlorothalonil, mancozeb, metalaxyl)",
            "Remove and destroy infected plants immediately",
            "Apply copper-based sprays as protective measure"
        ],
        "prevention": [
            "Plant resistant varieties",
            "Avoid overhead watering",
            "Space plants adequately for air circulation",
            "Monitor during cool wet weather"
        ]
    },
    "Tomato___Leaf_Mold": {
        "display_name": "Tomato Leaf Mold",
        "crop": "Tomato",
        "description": "Caused by Passalora fulva, leaf mold produces pale green or yellowish spots on upper leaf surfaces and olive-green to gray mold on undersides.",
        "severity": "Moderate",
        "treatment": [
            "Apply fungicides (chlorothalonil, copper, or mancozeb)",
            "Improve air circulation in greenhouse environments",
            "Remove infected leaves promptly"
        ],
        "prevention": [
            "Maintain relative humidity below 85%",
            "Increase plant spacing and pruning",
            "Use resistant varieties",
            "Avoid wetting leaves during irrigation"
        ]
    },
    "Tomato___Septoria_leaf_spot": {
        "display_name": "Tomato Septoria Leaf Spot",
        "crop": "Tomato",
        "description": "Caused by Septoria lycopersici, producing small circular spots with dark borders and gray-white centers on lower leaves first.",
        "severity": "Moderate",
        "treatment": [
            "Apply fungicides (chlorothalonil, copper, or mancozeb) at first sign",
            "Remove infected lower leaves",
            "Apply weekly during wet weather"
        ],
        "prevention": [
            "Avoid working with wet plants",
            "Mulch soil to prevent spore splash",
            "Rotate crops with non-solanaceous plants",
            "Remove plant debris at season end"
        ]
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "display_name": "Tomato Spider Mite Damage",
        "crop": "Tomato",
        "description": "Two-spotted spider mites (Tetranychus urticae) cause stippling, bronzing, or yellowing of leaves. Severe infestations cause leaf drop.",
        "severity": "Moderate to High",
        "treatment": [
            "Apply miticides (abamectin, bifenazate, or spiromesifen)",
            "Use neem oil or insecticidal soap as organic alternative",
            "Spray with forceful water jets to dislodge mites",
            "Apply predatory mites (Phytoseiidae) for biological control"
        ],
        "prevention": [
            "Monitor regularly especially during hot, dry weather",
            "Avoid excessive nitrogen fertilization",
            "Keep plants well-watered (stress increases susceptibility)",
            "Introduce natural predators"
        ]
    },
    "Tomato___Target_Spot": {
        "display_name": "Tomato Target Spot",
        "crop": "Tomato",
        "description": "Caused by Corynespora cassiicola, producing dark brown circular lesions with concentric rings (target spots) on leaves.",
        "severity": "Moderate",
        "treatment": [
            "Apply fungicides (azoxystrobin or difenoconazole) at early infection",
            "Remove severely infected leaves",
            "Improve air circulation around plants"
        ],
        "prevention": [
            "Avoid overhead irrigation",
            "Remove plant debris promptly",
            "Space plants properly for air circulation"
        ]
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "display_name": "Tomato Yellow Leaf Curl Virus",
        "crop": "Tomato",
        "description": "TYLCV is transmitted by whiteflies (Bemisia tabaci). Causes upward leaf curling, yellowing of leaf margins, and stunted plant growth.",
        "severity": "Critical",
        "treatment": [
            "No cure for infected plants — remove and destroy them",
            "Control whitefly vectors with imidacloprid or spinosad",
            "Use reflective aluminum mulches to repel whiteflies",
            "Apply systemic insecticides to protect remaining plants"
        ],
        "prevention": [
            "Plant TYLCV-resistant or tolerant varieties",
            "Use fine mesh net covers in nursery",
            "Control whitefly populations early in season",
            "Remove virus reservoir plants (weeds) from surrounding area"
        ]
    },
    "Tomato___Tomato_mosaic_virus": {
        "display_name": "Tomato Mosaic Virus",
        "crop": "Tomato",
        "description": "ToMV causes light and dark green mosaic patterns on leaves, distortion, and reduced fruit quality. Spreads through mechanical contact.",
        "severity": "High",
        "treatment": [
            "No cure — remove and destroy infected plants",
            "Disinfect tools with 10% bleach solution between plants",
            "Maintain biological control of aphid vectors"
        ],
        "prevention": [
            "Use ToMV-resistant varieties",
            "Wash hands thoroughly before handling plants",
            "Use certified virus-free seed",
            "Control aphids and other insect vectors"
        ]
    },
    "Tomato___healthy": {
        "display_name": "Healthy Tomato",
        "crop": "Tomato",
        "description": "The tomato leaf appears healthy with no visible signs of disease.",
        "severity": "None",
        "treatment": ["No treatment required"],
        "prevention": ["Continue regular monitoring", "Maintain balanced nutrition and irrigation"]
    },
    "Blueberry___healthy": {
        "display_name": "Healthy Blueberry",
        "crop": "Blueberry",
        "description": "The blueberry foliage exhibits vibrant green color and vigorous healthy growth with no pathogen symptoms.",
        "severity": "None",
        "treatment": ["No treatment required"],
        "prevention": [
            "Maintain soil pH between 4.5 and 5.2",
            "Apply acid-forming organic mulch such as pine bark",
            "Monitor periodically for insect vectors"
        ]
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "display_name": "Cherry Powdery Mildew",
        "crop": "Cherry",
        "description": "Caused by Podosphaera clandestina. Appears as circular powdery white fungal patches on leaves and young shoots, causing leaf curling and distortion.",
        "severity": "Moderate",
        "treatment": [
            "Apply sulfur-based fungicides or potassium bicarbonate sprays",
            "Prune infected twigs and leaves to promote canopy aeration",
            "Use myclobutanil or propiconazole at first sign of infection"
        ],
        "prevention": [
            "Avoid excessive nitrogen fertilization which produces susceptible tender growth",
            "Ensure adequate spacing and open tree canopies for air movement",
            "Apply protective fungicide sprays early in spring"
        ]
    },
    "Cherry_(including_sour)___healthy": {
        "display_name": "Healthy Cherry",
        "crop": "Cherry",
        "description": "The cherry foliage is healthy, showing smooth dark green leaves without mildew, spots, or discoloration.",
        "severity": "None",
        "treatment": ["No treatment required"],
        "prevention": [
            "Maintain balanced watering schedule",
            "Prune during dormant season to maintain good airflow",
            "Apply balanced fertilizer per soil test recommendations"
        ]
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "display_name": "Citrus Greening (Huanglongbing)",
        "crop": "Orange",
        "description": "Caused by Candidatus Liberibacter asiaticus, transmitted by Asian citrus psyllids. Produces asymmetrical blotchy mottling on leaves, vein yellowing, and stunted bitter fruit.",
        "severity": "Critical",
        "treatment": [
            "No known cure once a tree is infected; promptly eradicate infected trees to protect surrounding grove",
            "Control Asian citrus psyllid populations using targeted systemic insecticides (imidacloprid, thiamethoxam)",
            "Apply foliar nutritional sprays (zinc, manganese, iron) to support remaining tree vigor"
        ],
        "prevention": [
            "Plant only certified disease-free nursery stock",
            "Strictly monitor and control psyllid vector populations",
            "Establish regional quarantine zones to prevent pathogen transport"
        ]
    },
    "Peach___Bacterial_spot": {
        "display_name": "Peach Bacterial Spot",
        "crop": "Peach",
        "description": "Caused by Xanthomonas arboricola pv. pruni. Produces small angular water-soaked lesions that turn purple to dark brown and drop out, creating a 'shot-hole' appearance.",
        "severity": "High",
        "treatment": [
            "Apply copper-containing bactericides or oxytetracycline sprays beginning at petal fall",
            "Avoid overhead sprinkler irrigation that spreads bacterial exudate",
            "Prune and destroy infected cankers during winter dormancy"
        ],
        "prevention": [
            "Plant bacterial spot-resistant peach cultivars",
            "Avoid planting in excessively sandy sites with high wind exposure",
            "Maintain optimal soil fertility to prevent plant stress"
        ]
    },
    "Peach___healthy": {
        "display_name": "Healthy Peach",
        "crop": "Peach",
        "description": "The peach leaf is healthy with uniform green color and no bacterial lesions or fungal spotting.",
        "severity": "None",
        "treatment": ["No treatment required"],
        "prevention": [
            "Conduct routine scouting throughout the growing season",
            "Maintain good pruning practices for sunlight penetration",
            "Keep orchard floor clear of fallen debris"
        ]
    },
    "Pepper,_bell___Bacterial_spot": {
        "display_name": "Bell Pepper Bacterial Spot",
        "crop": "Pepper (Bell)",
        "description": "Caused by Xanthomonas campestris pv. vesicatoria. Produces small water-soaked lesions that turn dark brown with chlorotic yellow halos, causing extensive defoliation.",
        "severity": "High",
        "treatment": [
            "Apply fixed copper tank-mixed with mancozeb for enhanced bactericidal efficacy",
            "Remove and destroy severely blighted plants immediately",
            "Avoid handling foliage when wet to prevent mechanical transmission"
        ],
        "prevention": [
            "Use certified pathogen-free, hot-water treated seeds",
            "Rotate crops away from solanaceous plants (tomatoes, peppers, potatoes) for at least 2 years",
            "Use drip irrigation rather than overhead sprinklers"
        ]
    },
    "Pepper,_bell___healthy": {
        "display_name": "Healthy Bell Pepper",
        "crop": "Pepper (Bell)",
        "description": "The bell pepper leaf exhibits healthy turgor, uniform green pigmentation, and no necrotic spots or viral mottling.",
        "severity": "None",
        "treatment": ["No treatment required"],
        "prevention": [
            "Provide consistent moisture and well-draining soil",
            "Apply balanced nitrogen-phosphorus-potassium fertilizer",
            "Scout weekly for signs of insect pests"
        ]
    },
    "Raspberry___healthy": {
        "display_name": "Healthy Raspberry",
        "crop": "Raspberry",
        "description": "The raspberry cane and leaves appear robust, vibrant green, and free from fungal canes blight, rust, or viral yellows.",
        "severity": "None",
        "treatment": ["No treatment required"],
        "prevention": [
            "Prune out spent floricanes immediately after summer harvest",
            "Trellis canes to facilitate airflow and rapid drying",
            "Maintain 2-3 inches of mulch around base"
        ]
    },
    "Soybean___healthy": {
        "display_name": "Healthy Soybean",
        "crop": "Soybean",
        "description": "The trifoliate soybean leaf is fully expanded, uniformly green, and free from rust pustules, frogeye leaf spots, or bacterial blight.",
        "severity": "None",
        "treatment": ["No treatment required"],
        "prevention": [
            "Practice rotational cropping with non-host crops like corn",
            "Plant certified high-germination weed-free seed",
            "Monitor canopy closure and field moisture conditions"
        ]
    },
    "Squash___Powdery_mildew": {
        "display_name": "Squash Powdery Mildew",
        "crop": "Squash",
        "description": "Caused by Podosphaera xanthii. Forms white talcum-like fungal colonies on upper and lower leaf surfaces, causing premature senescence and sunburned fruit.",
        "severity": "Moderate",
        "treatment": [
            "Spray neem oil, horticultural oil, or potassium bicarbonate at first symptom onset",
            "Apply systemic fungicides (such as quinoxyfen or triflumizole) during heavy pressure",
            "Prune older heavily shaded leaves to increase sunlight exposure"
        ],
        "prevention": [
            "Select resistant squash and pumpkin varieties (PMR cultivars)",
            "Space plants widely to facilitate maximum airflow between rows",
            "Avoid excessive late-season nitrogen applications"
        ]
    },
    "Strawberry___Leaf_scorch": {
        "display_name": "Strawberry Leaf Scorch",
        "crop": "Strawberry",
        "description": "Caused by the fungus Diplocarpon earlianum. Symptoms include numerous small, irregular purple-to-brown spots that coalesce, causing leaf margins to curl and scorch.",
        "severity": "Moderate",
        "treatment": [
            "Apply captan, thiophanate-methyl, or copper fungicides at early flower stage",
            "Remove and compost/destroy severely scorched leaves during renovation",
            "Improve drainage and reduce planting density"
        ],
        "prevention": [
            "Plant scorch-resistant strawberry varieties",
            "Mow and renovate strawberry beds promptly after harvest",
            "Water with drip lines early in the morning"
        ]
    },
    "Strawberry___healthy": {
        "display_name": "Healthy Strawberry",
        "crop": "Strawberry",
        "description": "The strawberry foliage is dark green, vigorous, and shows no purple spots, powdery mildew, or necrotic leaf scorch.",
        "severity": "None",
        "treatment": ["No treatment required"],
        "prevention": [
            "Apply straw mulch to keep fruit and leaves off bare wet soil",
            "Ensure beds are well-drained with plenty of direct sunlight",
            "Perform post-harvest renovation annually"
        ]
    }
}


def get_disease_info(class_name: str) -> dict:
    """
    Get human-readable disease information and treatment recommendations for a predicted class.

    If the class_name is not in the database, returns a generic 'unknown' response.

    Args:
        class_name (str): Raw class name from the model (e.g., 'Tomato___Late_blight')

    Returns:
        dict: Disease information with display_name, crop, description, severity,
              treatment list, and prevention list
    """
    if class_name in DISEASE_INFO:
        return DISEASE_INFO[class_name]

    # Generic fallback for classes not explicitly defined in the dictionary
    parts = class_name.split('___')
    crop = parts[0].replace('_', ' ') if parts else 'Unknown'
    disease = parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown'
    is_healthy = 'healthy' in class_name.lower()

    return {
        "display_name": disease if not is_healthy else f"Healthy {crop}",
        "crop": crop,
        "description": f"This condition was detected in {crop} plants." if not is_healthy
                       else f"The {crop} leaf appears healthy.",
        "severity": "None" if is_healthy else "Consult an agronomist",
        "treatment": ["No treatment required"] if is_healthy
                     else ["Consult with local agricultural extension officer"],
        "prevention": ["Maintain good agricultural practices", "Regular monitoring recommended"]
    }


def extract_crop_type(class_name: str) -> str:
    """Extract crop name from class label (e.g., 'Tomato___Late_blight' → 'Tomato')."""
    parts = class_name.split('___')
    return parts[0].replace('_', ' ') if parts else 'Unknown'


def is_healthy_prediction(class_name: str) -> bool:
    """Check if a class name represents a healthy plant (no disease)."""
    return 'healthy' in class_name.lower()


def format_confidence(confidence: float) -> str:
    """Format confidence score as a readable percentage string."""
    return f"{confidence * 100:.1f}%"


def allowed_file(filename: str) -> bool:
    """
    Check if an uploaded file has an allowed image extension.

    Allowed extensions: .jpg, .jpeg, .png, .webp, .bmp

    Args:
        filename (str): Original filename from upload

    Returns:
        bool: True if extension is allowed
    """
    allowed_extensions = {'jpg', 'jpeg', 'png', 'webp', 'bmp'}
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in allowed_extensions


def generate_unique_filename(original_filename: str) -> str:
    """
    Generate a unique filename for uploaded images to prevent collisions.
    Format: YYYYMMDD_HHMMSS_<random6chars>.<ext>

    Args:
        original_filename (str): Original uploaded filename

    Returns:
        str: Unique filename with original extension
    """
    import uuid
    ext = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else 'jpg'
    unique_id = uuid.uuid4().hex[:6]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{timestamp}_{unique_id}.{ext}"


def validate_registration_input(username: str, email: str, password: str) -> list:
    """
    Validate user registration input. Returns a list of error messages (empty if valid).

    Validation rules:
        - Username: 3-50 characters, alphanumeric and underscore only
        - Email: must contain @ and valid domain
        - Password: minimum 6 characters

    Args:
        username, email, password (str): Input values

    Returns:
        list: List of error message strings (empty list = all valid)
    """
    errors = []

    if not username or len(username) < 3:
        errors.append("Username must be at least 3 characters long")
    elif len(username) > 50:
        errors.append("Username must not exceed 50 characters")
    elif not re.match(r'^[a-zA-Z0-9_]+$', username):
        errors.append("Username may only contain letters, numbers, and underscores")

    if not email or '@' not in email or '.' not in email.split('@')[-1]:
        errors.append("Please provide a valid email address")

    if not password or len(password) < 6:
        errors.append("Password must be at least 6 characters long")

    return errors
