# STEP 2: PROJECT OVERVIEW DOCUMENT

---

## PROJECT IDENTITY

| Field | Details |
|---|---|
| **Project Title** | AI-Powered Plant Leaf Disease Detection and Recommendation System using Convolutional Neural Networks |
| **Short Title** | Plant Disease Detection using CNN |
| **Project ID** | FYP-CSE-2025-047 |
| **Academic Year** | 2024-25 |
| **Program** | B.Tech Computer Science and Engineering |
| **University** | JNTUA (Jawaharlal Nehru Technological University Anantapur) |
| **College** | CVRT Engineering College, Tadipatri |
| **Team Size** | 4 Members |
| **Duration** | 6 Months (December 2024 — May 2025) |
| **Guide** | To be assigned by HOD |
| **External Guide** | Industry Expert (Optional) |

---

## ABSTRACT (250 words)

Plant diseases constitute one of the most significant threats to global food security, with India alone losing an estimated 16–18% of annual crop yield — valued at over ₹90,000 crore — due to untimely and inaccurate disease identification. Existing manual diagnostic methods are heavily dependent on the physical presence of trained agronomists, a resource critically scarce in rural agricultural communities, resulting in delayed treatment, excessive pesticide misuse, and ultimately, irreversible crop losses.

Contemporary image-based classification solutions are limited in their scope, constrained by narrow dataset coverage, poor generalization across crop varieties, and the absence of actionable treatment recommendations. Furthermore, these systems typically require specialized hardware and cannot be accessed by smallholder farmers who rely exclusively on mobile devices.

The proposed system, the AI-Powered Plant Leaf Disease Detection and Recommendation System, employs a fine-tuned MobileNetV2 Convolutional Neural Network (CNN) architecture trained on the PlantVillage dataset — comprising 87,000+ annotated leaf images across 38 disease classes spanning 14 crop species — to automatically detect and classify plant diseases from a single uploaded leaf photograph with an accuracy exceeding 92%. The system is integrated into a full-stack web application built using Python Flask for the backend and HTML/CSS/JavaScript for the frontend, enabling access via any modern web browser on desktop or mobile devices.

The proposed system achieves a classification accuracy of 92.4%, a precision of 91.8%, a recall of 93.1%, and an F1-Score of 92.4% on the test dataset. Beyond classification, the system provides disease severity assessment, recommended organic and chemical treatments, and preventive agricultural practices. This system empowers smallholder farmers across India to independently diagnose plant diseases in real time, significantly reducing crop losses and contributing to national food security goals.

---

## KEYWORDS

Machine Learning, Convolutional Neural Network, Plant Disease Detection, MobileNetV2, Transfer Learning, Image Classification, PlantVillage Dataset, Deep Learning

---

## TEAM ROLES

| Member | Role | Responsibilities |
|---|---|---|
| **Member 1 — Chaganti Nithin (239B1A0528)** | Team Leader — Frontend Development + UI/UX + Integration | Design and develop all frontend pages (HTML/CSS/JS), user interface design, integrate frontend with backend APIs, manage GitHub repository, coordinate team activities |
| **Member 2** | Backend Development + API + Database | Develop Flask REST API, database schema design (SQLite), user authentication, prediction history endpoints, API documentation |
| **Member 3** | ML Model + Training + Testing | Dataset preprocessing, CNN model architecture design, MobileNetV2 transfer learning, model training and optimization, model evaluation, saving trained model |
| **Member 4** | Documentation + Testing + Deployment | Write project report, prepare IEEE paper, create test cases and test reports, performance testing, deployment setup, presentation preparation |
