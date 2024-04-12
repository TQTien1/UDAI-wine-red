import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,f1_score,precision_score,recall_score
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt #visualization
import seaborn as sns #visualization
from model import new_model

st.set_page_config(layout="wide")
st.title("k-Nearest Neighbour Model")
st.set_option('deprecation.showPyplotGlobalUse', False)
col1, col2 = st.columns([0.5,0.5])

def drop_dupe():
    wine.drop_duplicates(inplace=True)
    st.write("Dữ liệu trùng sau khi xóa: ", wine[wine.columns.tolist()].duplicated().sum())

with st.container():
    df = st.sidebar.file_uploader("Upload Dataset", type='csv')
if df is not None:
    wine = pd.read_csv(df)
    with st.container(): #Bỏ container maybe
        func = st.sidebar.radio('Graphs', options=["Kiểm tra dữ liệu","Giá trị trung bình mỗi thuộc tính","Phân bố mỗi thuộc tính", "Biểu đồ độ tương quan và Dự đoán chất lượng rượu"])

    with st.container():
        if func == "Kiểm tra dữ liệu":
            with col1: 
                st.write("Số dòng: ", wine.shape[0])
                st.write("Số cột: ", wine.shape[1], "\n")
                #Kiểm Tra dữ liệu thiếu
                st.write("Dữ liệu  trùng: ", wine[wine.columns.tolist()].duplicated().sum())
                if st.button("Xóa dữ liệu thiếu"):
                    drop_dupe()

            with col2:
                st.write("Missing values:")
                st.write(wine[wine.columns.tolist()].isnull().sum())


    with st.container():
        if func == "Giá trị trung bình mỗi thuộc tính":
            attributes = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',	'total sulfur dioxide',	'density','pH','sulphates','alcohol', 'quality']
            mean_values = wine[attributes].mean()
            plt.figure(figsize = (15,7))
            mean_values.plot(kind = 'bar')
            plt.title('Giá trị trung bình các thuộc tính')
            plt.xlabel('Thuộc tính')
            plt.ylabel('Giá trị')
            plt.xticks(rotation = 45)
            st.pyplot(plt)

    with st.container():
        if func == "Phân bố mỗi thuộc tính":
            plt.figure(figsize=(13,6))
            plt.subplot(121)
            wine["quality"].value_counts().plot.pie(autopct  = "%1.0f%%",
                                                    fontsize = 13,
                                                    wedgeprops = {"linewidth" : 2,
                                                                "edgecolor" : "w"},
                                                )
            plt.title("Phần trăm quality trong bộ dữ liệu")
            plt.ylabel("")
            plt.subplot(122)
            ax = sns.countplot(y = wine["quality"],linewidth = 2,
                            edgecolor = "k")
            for i,j in enumerate(wine["quality"].value_counts().values) :
                if i == 0:
                    ax.text(.1,2,j,fontsize = 15,color = "k")  # Fix: Use "black" instead of "k"
                if i ==1:
                    ax.text(.1,3,j,fontsize = 15,color = "k")
                if i == 2:
                    ax.text(.1,4,j,fontsize = 15,color = "k")
                if i == 3:
                    ax.text(.1,1,j,fontsize = 15,color = "k")
                if i == 4:
                    ax.text(.1,5,j,fontsize = 15,color = "k")
                if i == 5:
                    ax.text(.1,0,j,fontsize = 15,color = "k")
            plt.title("Tổng số lượng quality")
            plt.grid(True,alpha = .1)
            st.pyplot(plt)

    with st.container():
        with st.container():
            if func == "Biểu đồ độ tương quan và Dự đoán chất lượng rượu":
                correlation=wine.corr()
                print(correlation['quality'].sort_values(ascending=False))
                correlation['quality'].drop('quality').sort_values(ascending=False).plot(kind='bar',figsize=(10,5))
                plt.title("Biểu đồ độ tương quan")
                st.pyplot(plt)
                with st.form("Loại bỏ cột"):
                    st.write("Loại bỏ cột")
                    a = st.checkbox('fixed acidity')
                    b = st.checkbox('volatile acidity')
                    c = st.checkbox('citric acid')
                    d = st.checkbox('residual sugar')
                    e = st.checkbox('chlorides')
                    f = st.checkbox('free sulfur dioxide')
                    g = st.checkbox('total sulfur dioxide')
                    h = st.checkbox('density')
                    i = st.checkbox('pH')
                    k = st.checkbox('sulphates')
                    l = st.checkbox('alcohol')
                    submitted = st.form_submit_button()
                    if submitted:
                        if a: wine = wine.drop('fixed acidity', axis = 1)
                        if b: wine = wine.drop('volatile acidity', axis = 1)
                        if c: wine = wine.drop('citric acid', axis = 1)
                        if d: wine = wine.drop('residual sugar', axis = 1)
                        if e: wine = wine.drop('chlorides', axis = 1)
                        if f: wine = wine.drop('free sulfur dioxide', axis = 1)
                        if g: wine = wine.drop('total sulfur dioxide', axis = 1)
                        if h: wine = wine.drop('density', axis = 1)
                        if i: wine = wine.drop('pH', axis = 1)
                        if k: wine = wine.drop('sulphates', axis = 1)
                        if l: wine = wine.drop('alcohol', axis = 1)
                    st.write(wine)

            with st.form("Predict"):
                if 'fixed acidity' in wine:
                    fixed_acidity = st.number_input('fixed acidity')
                if 'volatile acidity' in wine:
                    volatile_acidity = st.number_input('volatile acidity')
                if 'citric acid' in wine:
                    citric_acid = st.number_input('citric acid')
                if 'residual sugar' in wine:
                    residual_sugar = st.number_input('residual sugar')
                if 'chlorides' in wine:
                    chlorides = st.number_input('chlorides')
                if 'free sulfur dioxide' in wine:
                    free_sulfur_dioxide = st.number_input('free sulfur dioxide')
                if 'total sulfur dioxide' in wine:
                    total_sulfur_dioxide = st.number_input('total sulfur dioxide')
                if 'density' in wine:
                    density = st.number_input('density')
                if 'pH' in wine:
                    pH = st.number_input('pH')
                if 'sulphates' in wine:
                    sulphates = st.number_input('sulphates')
                if 'alcohol' in wine:
                    alcohol = st.number_input('alcohol')
                submitted = st.form_submit_button()
                if submitted: 
                    user_X_test = [[fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol]]
                    user_pred = new_model.predict(user_X_test)
                    st.write("Quality dự đoán: ",user_pred)
