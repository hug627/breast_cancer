import streamlit as st   

st.set_page_config(
    page_title="Breast Cancer Prediction Health System",
)

st.header("Breast Cancer Prediction Health System")

st.subheader("📋 Objective of This Project")
st.write("""
         
Breast Cancer is the most common cancer amongst women in the world. It affects millions of people in 
the world . """)
st.write("""It starts when cells in the breast begin to grow out of control. These cells are usually tumors  that can be seen 
via X-ray or felt lumps in the breast area. """)
st.write("""The key challenge aganist its detection is how to classify tumors into 
         
    i. Malignant which is Cancerous 
    
    ii. Benign which is Non-Cancerous """)
st.write("""Develop a machine learning prediction system that identifies whether a patient has breast cancer.""")

st.write("""The objective will be achieved through:
""")

st.subheader("1️⃣ Data Understanding")
st.write("""
This project uses the 
   1. **Breast Cancer Dataset**
""")
st.subheader("2️⃣ Main Features Used")
st.write("""
The model uses **key features** to predict breast cancer risk:
   1. id: A unique identifier for each tumor record 
   2. Diagnosis: The diagnosis of the tumor where M stands for Malignant and B stands for Benign
   3. Radius_mean: The mean of the distances from the center of the tumor to the perimeter. It reflects the average size of the tumor 
   4. Texture_mean: The mean of the standard deviation of gray-scale values in thee tumor image. It measures the smoothness or coarseness of the tumor's texture. 
   5. Perimeter_mean: The mean of the perimeter lengths of the tumor's boundary. It reflects the average circumference of the tumor
   6. Area_mean: The mean of the area occupied by the tumor. It represents the average size of the tumor's surface area 
   7. Smoothness_mean: The mean of local variation in radius lengths, measuring how smooth the tumor's surface is
   8. Compactness_mean: The mean of the compactness of the tumor. It indicates how compact or round the tumor is
   9. Concavity_mean: The mean of the severity of concave portions of the tumor's boundary. It reflects the degree of inward curving of the tumor's edges
   10. Concave points_mean: The mean of the number of concave portions of the tumor's boundary. It counts the number of indentations or concave points on the tumor's edges
   11. Symmetry_mean: The mean of the symmetry of the tumor. It measures how symmetrical the tumor is about its center
   12. Fractal_dimension_mean: The mean of the fractal dimension reflecting how detailed the tumor's surface is
   13. Radius_se: The standard error of the radius measurement 
   14. Texture_se: The standard error of the texture measurement
   15. Perimeter_se: The standard error of the perimeter measurement 
   16. Area_se: The standard error of the area measurement 
   17. Smoothness_se: The standard error of the smoothness measurement
   18. Compactness_se: The standard error of the compactness measurement
   19. Concavity_se: The standard error of the concavity measurement 
   20. Concave_points_se: The Standard error of the concave points measurements
   21. Symmetry_se: The standard error of the symmetry measurement
   22. Fractal_dimension_se: The standard error of the fractal dimension measurement
   23. Radius _worst: The worst(largest) value of the radius measurements across all the samples 
   24. Texture_worst: The worst(largest) value of the texture measurements across all samples 
   25. Perimeter_worst: The worst(largest) value of the perimeter measurements 
   26. Area_worst: The worst(largest) value of the area measurements across all the samples 
   27. Smoothness_worst: The worst(largest) value of the smoothness measurements across all the samples 
   28. Compactness_worst: The worst(largest) value of the compactness measurements across all the sample 
   29. Concavity_worst: The worst(largest) value of the concavity measurements across all the samples 
   30. Concave points_worst: The worst(largest) value of the concave points measurements across all the samples 
   31. Symmetry_worst: The worst(largest) value of the symmetry across all the samples 
   32. Fractal_dimension_worst: The worst(largest) value of the fractal dimension measurements across all the samples 
         """)

st.write("""
         Basically the numerous measurements have been made for each single tumor, and in this dataset they chose to show the average measurements, the worst(biggest) one and the squared error. 
         """)


st.subheader("3️⃣ How It Works")
st.write("""
The system uses advanced machine learning algorithms trained on historical patient data to:

1. **Analyze Input Features**
2. **Calculate Risk Score**
3. **Provide Prediction** 
4. **Recommend Actions** 

**Machine Learning Models Used:**
- Support Vector Machines (SVM)
- Logistic Regression
- Random Forest
- XGBoost
- Neural Networks (MLP)

The models are optimized to maximize cancer detection rate (recall) while maintaining 
reasonable precision to minimize false alarms.
""")

st.subheader("4️⃣ Expected Impact")
st.write("""
By implementing this breast cancer prediction system, we aim to:

🎯 **Improve Early Detection:** Identify high-risk patients before symptoms become severe

💰 **Reduce Healthcare Costs:** Provide cost-effective preliminary screening before expensive tests

⏱️ **Save Time:** Quickly assess large populations and prioritize those needing immediate attention

📊 **Support Research:** Provide insights into lung cancer risk factors and symptom patterns

❤️ **Save Lives:** Enable earlier intervention and treatment, improving patient outcomes

This project demonstrates how machine learning can support public health initiatives and 
assist medical professionals in making data-driven decisions for better patient care.
""")

# Optional: Add a disclaimer section
st.markdown("---")
st.warning("""
⚠️ **Medical Disclaimer:**

This prediction system is for educational and screening purposes only. It does not provide 
medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals 
for medical decisions. If you suspect you have breast cancer or experience concerning symptoms, 
seek immediate medical attention.
""")

st.info("""
ℹ️ **About This Project:**

This is a machine learning project developed to demonstrate the application of AI in healthcare. 
The models are trained on survey data and should be validated with medical professionals before 
any clinical use.
""")