import os
HF_TOKEN = os.getenv("HF_TOKEN")

import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
from math import sqrt
from scipy import stats as st
from matplotlib import pyplot as plt

from sklearn.calibration import CalibratedClassifierCV

import shap
import gradio as gr
import random
import re
import textwrap
from datasets import load_dataset


#Read data training data.

x1 = load_dataset("mertkarabacak/NCDB-GBM", data_files="6m_data_train.csv", use_auth_token = HF_TOKEN)
x1 = pd.DataFrame(x1['train'])
x1 = x1.iloc[:, 1:]

x2 = load_dataset("mertkarabacak/NCDB-GBM", data_files="12m_data_train.csv", use_auth_token = HF_TOKEN)
x2 = pd.DataFrame(x2['train'])
x2 = x2.iloc[:, 1:]

x3 = load_dataset("mertkarabacak/NCDB-GBM", data_files="18m_data_train.csv", use_auth_token = HF_TOKEN)
x3 = pd.DataFrame(x3['train'])
x3 = x3.iloc[:, 1:]

x4 = load_dataset("mertkarabacak/NCDB-GBM", data_files="24m_data_train.csv", use_auth_token = HF_TOKEN)
x4 = pd.DataFrame(x4['train'])
x4 = x4.iloc[:, 1:]

#Read validation data.

x1_valid = load_dataset("mertkarabacak/NCDB-GBM", data_files="6m_data_valid.csv", use_auth_token = HF_TOKEN)
x1_valid = pd.DataFrame(x1_valid['train'])
x1_valid = x1_valid.iloc[:, 1:]

x2_valid = load_dataset("mertkarabacak/NCDB-GBM", data_files="12m_data_valid.csv", use_auth_token = HF_TOKEN)
x2_valid = pd.DataFrame(x2_valid['train'])
x2_valid = x2_valid.iloc[:, 1:]

x3_valid = load_dataset("mertkarabacak/NCDB-GBM", data_files="18m_data_valid.csv", use_auth_token = HF_TOKEN)
x3_valid = pd.DataFrame(x3_valid['train'])
x3_valid = x3_valid.iloc[:, 1:]

x4_valid = load_dataset("mertkarabacak/NCDB-GBM", data_files="24m_data_valid.csv", use_auth_token = HF_TOKEN)
x4_valid = pd.DataFrame(x4_valid['train'])
x4_valid = x4_valid.iloc[:, 1:]


#Define feature names.
f1_names = list(x1.columns)
f1_names = [f1.replace('__', ' - ') for f1 in f1_names]
f1_names = [f1.replace('_', ' ') for f1 in f1_names]

f2_names = list(x2.columns)
f2_names = [f2.replace('__', ' - ') for f2 in f2_names]
f2_names = [f2.replace('_', ' ') for f2 in f2_names]

f3_names = list(x3.columns)
f3_names = [f3.replace('__', ' - ') for f3 in f3_names]
f3_names = [f3.replace('_', ' ') for f3 in f3_names]

f4_names = list(x4.columns)
f4_names = [f4.replace('__', ' - ') for f4 in f4_names]
f4_names = [f4.replace('_', ' ') for f4 in f4_names]


#Prepare training data for the outcome 1.
y1= x1.pop('OUTCOME')

#Prepare training data for the outcome 2.
y2 = x2.pop('OUTCOME')

#Prepare training data for the outcome 3.
y3 = x3.pop('OUTCOME')

#Prepare training data for the outcome 4.
y4 = x4.pop('OUTCOME')

#Prepare validation data for the outcome 1.
y1_valid = x1_valid.pop('OUTCOME')

#Prepare validation data for the outcome 2.
y2_valid = x2_valid.pop('OUTCOME')

#Prepare validation data for the outcome 3.
y3_valid = x3_valid.pop('OUTCOME')

#Prepare validation data for the outcome 4.
y4_valid = x4_valid.pop('OUTCOME')

#Training models.

from tabpfn import TabPFNClassifier
tabpfn = TabPFNClassifier(device='cuda', N_ensemble_configurations=1)

y1_model = tabpfn
y1_model = y1_model.fit(x1, y1, overwrite_warning=True)

y1_calib_model = CalibratedClassifierCV(y1_model, method='isotonic', cv='prefit')
y1_calib_model = y1_calib_model.fit(x1_valid, y1_valid)

y1_explainer = shap.Explainer(y1_model.predict, x1)


from tabpfn import TabPFNClassifier
tabpfn = TabPFNClassifier(device='cuda', N_ensemble_configurations=1)

y2_model = tabpfn
y2_model = y2_model.fit(x2, y2, overwrite_warning=True)

y2_calib_model = CalibratedClassifierCV(y2_model, method='isotonic', cv='prefit')
y2_calib_model = y2_calib_model.fit(x2_valid, y2_valid)

y2_explainer = shap.Explainer(y2_model.predict, x2)


from tabpfn import TabPFNClassifier
tabpfn = TabPFNClassifier(device='cuda', N_ensemble_configurations=1)

y3_model = tabpfn
y3_model = y3_model.fit(x3, y3, overwrite_warning=True)

y3_calib_model = CalibratedClassifierCV(y3_model, method='isotonic', cv='prefit')
y3_calib_model = y3_calib_model.fit(x3_valid, y3_valid)

y3_explainer = shap.Explainer(y3_model.predict, x3)


from tabpfn import TabPFNClassifier
tabpfn = TabPFNClassifier(device='cuda', N_ensemble_configurations=1)

y4_model = tabpfn
y4_model = y4_model.fit(x4, y4, overwrite_warning=True)

y4_calib_model = CalibratedClassifierCV(y4_model, method='isotonic', cv='prefit')
y4_calib_model = y4_calib_model.fit(x4_valid, y4_valid)

y4_explainer = shap.Explainer(y4_model.predict, x4)


output_y1 = (
    """          
        <br/>
        <center>The probability of 6-month survival:</center>
        <br/>
        <center><h1>{:.2f}%</h1></center>
"""
)

output_y2 = (
    """          
        <br/>        
        <center>The probability of 12-month survival:</center>
        <br/>        
        <center><h1>{:.2f}%</h1></center>
"""
)

output_y3 = (
    """          
        <br/>        
        <center>The probability of 18-month survival:</center>
        <br/>        
        <center><h1>{:.2f}%</h1></center>
"""
)

output_y4 = (
    """          
        <br/>        
        <center>The probability of 24-month survival:</center>
        <br/>        
        <center><h1>{:.2f}%</h1></center>
"""
)


#Define predict for y1.
def y1_predict(*args):
    df1 = pd.DataFrame([args], columns=x1.columns)
    pos_pred = y1_calib_model.predict_proba(df1)
    prob = pos_pred[0][1]
    prob = 1-prob
    output = output_y1.format(prob * 100)
    return output

#Define predict for y2.
def y2_predict(*args):
    df2 = pd.DataFrame([args], columns=x2.columns)
    pos_pred = y2_calib_model.predict_proba(df2)
    prob = pos_pred[0][1]
    prob = 1-prob
    output = output_y2.format(prob * 100)
    return output

#Define predict for y3.
def y3_predict(*args):
    df3 = pd.DataFrame([args], columns=x3.columns)
    pos_pred = y3_calib_model.predict_proba(df3)
    prob = pos_pred[0][1]
    prob = 1-prob    
    output = output_y3.format(prob * 100)
    return output

#Define predict for y4.
def y4_predict(*args):
    df4 = pd.DataFrame([args], columns=x4.columns)
    pos_pred = y4_calib_model.predict_proba(df4)
    prob = pos_pred[0][1]
    prob = 1-prob    
    output = output_y4.format(prob * 100)
    return output


#Define function for wrapping feature labels.
def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_yticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width, break_long_words=break_long_words))
    ax.set_yticklabels(labels, rotation=0)
    

#Define interpret for y1.
def y1_interpret(*args):
    df1 = pd.DataFrame([args], columns=x1.columns)
    shap_values1 = y1_explainer(df1).values
    shap_values1 = np.abs(shap_values1)
    shap.bar_plot(shap_values1[0], max_display = 10, show = False, feature_names = f1_names)
    fig = plt.gcf()
    ax = plt.gca()
    wrap_labels(ax, 20)
    ax.figure
    plt.tight_layout()
    fig.set_figheight(7)
    fig.set_figwidth(9)
    plt.xlabel("SHAP value (impact on model output)", fontsize =12, fontweight = 'heavy', labelpad = 8)
    plt.tick_params(axis="y",direction="out", labelsize = 12)
    plt.tick_params(axis="x",direction="out", labelsize = 12)
    return fig

#Define interpret for y2.
def y2_interpret(*args):
    df2 = pd.DataFrame([args], columns=x2.columns)
    shap_values2 = y2_explainer(df2).values
    shap_values2 = np.abs(shap_values2)
    shap.bar_plot(shap_values2[0], max_display = 10, show = False, feature_names = f2_names)
    fig = plt.gcf()
    ax = plt.gca()
    wrap_labels(ax, 20)
    ax.figure
    plt.tight_layout()
    fig.set_figheight(7)
    fig.set_figwidth(9)
    plt.xlabel("SHAP value (impact on model output)", fontsize =12, fontweight = 'heavy', labelpad = 8)
    plt.tick_params(axis="y",direction="out", labelsize = 12)
    plt.tick_params(axis="x",direction="out", labelsize = 12)
    return fig

#Define interpret for y3.
def y3_interpret(*args):
    df3 = pd.DataFrame([args], columns=x3.columns)
    shap_values3 = y3_explainer(df3).values
    shap_values3 = np.abs(shap_values3)
    shap.bar_plot(shap_values3[0], max_display = 10, show = False, feature_names = f3_names)
    fig = plt.gcf()
    ax = plt.gca()
    wrap_labels(ax, 20)
    ax.figure
    plt.tight_layout()
    fig.set_figheight(7)
    fig.set_figwidth(9)
    plt.xlabel("SHAP value (impact on model output)", fontsize =12, fontweight = 'heavy', labelpad = 8)
    plt.tick_params(axis="y",direction="out", labelsize = 12)
    plt.tick_params(axis="x",direction="out", labelsize = 12)
    return fig

#Define interpret for y4.
def y4_interpret(*args):
    df4 = pd.DataFrame([args], columns=x4.columns)
    shap_values4 = y4_explainer(df4).values
    shap_values4 = np.abs(shap_values4)
    shap.bar_plot(shap_values4[0], max_display = 10, show = False, feature_names = f4_names)
    fig = plt.gcf()
    ax = plt.gca()
    wrap_labels(ax, 20)
    ax.figure
    plt.tight_layout()
    fig.set_figheight(7)
    fig.set_figwidth(9)
    plt.xlabel("SHAP value (impact on model output)", fontsize =12, fontweight = 'heavy', labelpad = 8)
    plt.tick_params(axis="y",direction="out", labelsize = 12)
    plt.tick_params(axis="x",direction="out", labelsize = 12)
    return fig


with gr.Blocks(title = "NCDB-Meningioma") as demo:
        
    gr.Markdown(
        """
    <br/>
    <center><h2>NOT FOR CLINICAL USE</h2><center>    
    <br/>    
    <center><h1>IDH-wt Glioblastoma Survival Outcomes</h1></center>
    <center><h2>Prediction Tool</h2></center>
    <br/>
    <center><h3>This web application should not be used to guide any clinical decisions.</h3><center>
    <br/>
    <center><i>The publication describing the details of this prediction tool will be posted here upon the acceptance of publication.</i><center>
        """
    )

    gr.Markdown(
        """
        <center><h3>Model Performances</h3></center>
          <div style="text-align:center;">
          <table style="width:100%;">
          <tr>
            <th>Outcome</th>
            <th>Algorithm</th>
            <th>Sensitivity</th>
            <th>Specificity</th>
            <th>Accuracy</th>
            <th>AUPRC</th>
            <th>AUROC</th>
            <th>Brier Score</th>
          </tr>
          <tr>
            <td>6-Month Mortality</td>
            <td>TabPFN</td>
            <td>0.782 (0.761 - 0.803)</td>
            <td>0.740 (0.717 - 0.763)</td>
            <td>0.751 (0.729 - 0.773)</td>
            <td>0.647 (0.622 - 0.672)</td>
            <td>0.836 (0.805 - 0.853)</td>
            <td>0.135 (0.117 - 0.153)</td>             
          </tr>
          <tr>
            <td>12-Month Mortality</td>
            <td>TabPFN</td>
            <td>0.758 (0.736 - 0.780)</td>
            <td>0.626 (0.601 - 0.651)</td>
            <td>0.689 (0.665 - 0.713)</td>
            <td>0.740 (0.717 - 0.763)</td>
            <td>0.780 (0.748 - 0.795)</td>
            <td>0.196 (0.175 - 0.217)</td>             
          </tr>
          <tr>
            <td>18-Month Mortality</td>
            <td>TabPFN</td>
            <td>0.662 (0.637 - 0.687)</td>
            <td>0.657 (0.632 - 0.682)</td>
            <td>0.660 (0.635 - 0.685)</td>
            <td>0.809 (0.788 - 0.830)</td>
            <td>0.732 (0.695 - 0.750)</td>
            <td>0.200 (0.179 - 0.221)</td>             
          </tr>
          <tr>
            <td>24-Month Mortality</td>
            <td>TabPFN</td>
            <td>0.667 (0.641 - 0.693)</td>
            <td>0.626 (0.600 - 0.652)</td>
            <td>0.658 (0.632 - 0.684)</td>
            <td>0.882 (0.864 - 0.900)</td>
            <td>0.724 (0.683 - 0.749)</td>
            <td>0.154 (0.134 - 0.174)</td>             
          </tr>
        </table>
        </div>
        """
    )    

    with gr.Row():

        with gr.Column():

            Age = gr.Slider(label="Age", minimum = 18, maximum = 99, step = 1, value = 55)

            Sex = gr.Dropdown(label = "Sex", choices = ['Male', 'Female'], type = 'index', value = 'Male')
            
            Race = gr.Dropdown(label = "Race", choices = ['White', 'Black', 'Other'], type = 'index', value = 'White')

            Hispanic_Ethnicity = gr.Dropdown(label = "Hispanic Ethnicity", choices = ['No', 'Yes', 'Unknown'], type = 'index', value = 'No')
            
            Insurance_Status = gr.Dropdown(label = "Insurance Status", choices = ['Private insurance', 'Medicare', 'Medicaid', 'Other government', 'Not insured', 'Unknown'], type = 'index', value = 'Private insurance')
            
            Facility_Type = gr.Dropdown(label = "Facility Type", choices = ['Academic/Research Program', 'Community Cancer Program', 'Integrated Network Cancer Program'], type = 'index', value = 'Academic/Research Program')
            
            Facility_Location = gr.Dropdown(label = "Facility Location", choices = ['Central', 'Atlantic', 'Pacific', 'Mountain', 'New England'], type = 'index', value = 'Central')

            CharlsonDeyo_Score = gr.Dropdown(label = "Charlson-Deyo Score", choices = ['0', '1', '>2'], type = 'index', value = '0')
            
            MGMT_Methylation = gr.Dropdown(label = "MGMT Methylation", choices = ['Unmethylated', 'Methylated'], type = 'index', value = 'Unmethylated')
            
            Tumor_Size = gr.Slider(label = "Tumor Size (mm)", minimum = 1, maximum = 300, step = 1, value = 30)
            
            Extent_of_Resection = gr.Dropdown(label = 'Extent of Resection', choices = ['No resective surgery was performed', 'Gross total resection', 'Subtotal resection'], type = 'index', value = 'Gross total resection')
            
            Radiotherapy = gr.Dropdown(label = 'Radiotherapy', choices = ['No', 'Yes'], type = 'index', value = 'Yes')
            
            Chemotherapy = gr.Dropdown(label = "Chemotherapy", choices = ['No', 'Yes'], type = 'index', value = 'Yes')
            
            Immunotherapy = gr.Dropdown(label = "Immunotherapy", choices = ['No', 'Yes'], type = 'index', value = 'No')
            
        with gr.Column():
            
            with gr.Box():
                
                gr.Markdown(
                    """
                    <center> <h2>6-Month Survival</h2> </center>
                    <br/>
                    <center> This model uses the TabPFN algorithm.</center>
                    <br/>
                    """
                    )
                
                with gr.Row():
                    y1_predict_btn = gr.Button(value="Predict")
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                label1 = gr.Markdown()
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                with gr.Row():
                    y1_interpret_btn = gr.Button(value="Explain")
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                plot1 = gr.Plot()
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
            with gr.Box():
                gr.Markdown(
                    """
                    <center> <h2>12-Month Survival</h2> </center>
                    <br/>
                    <center> This model uses the TabPFN algorithm.</center>
                    <br/>
                    """
                    )
                
                with gr.Row():
                    y2_predict_btn = gr.Button(value="Predict")
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                label2 = gr.Markdown()
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                with gr.Row():
                    y2_interpret_btn = gr.Button(value="Explain")
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                plot2 = gr.Plot()
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
            with gr.Box():
                
                gr.Markdown(
                    """
                    <center> <h2> 18-Month Survival</h2> </center>
                    <br/>
                    <center> This model uses the TabPFN algorithm.</center>
                    <br/>
                    """
                    )
                
                with gr.Row():
                    y3_predict_btn = gr.Button(value="Predict")
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                label3 = gr.Markdown()
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                with gr.Row():
                    y3_interpret_btn = gr.Button(value="Explain")
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                plot3 = gr.Plot()
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )  

            with gr.Box():
                
                gr.Markdown(
                    """
                    <center> <h2> 24-Month Survival</h2> </center>
                    <br/>
                    <center> This model uses the TabPFN algorithm.</center>
                    <br/>
                    """
                    )
                
                with gr.Row():
                    y4_predict_btn = gr.Button(value="Predict")
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                label4 = gr.Markdown()
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                with gr.Row():
                    y4_interpret_btn = gr.Button(value="Explain")
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                plot4 = gr.Plot()
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )                
           
                y1_predict_btn.click(
                    y1_predict,
                    inputs = [Age, Sex, Race, Hispanic_Ethnicity, Insurance_Status, Facility_Type, Facility_Location, CharlsonDeyo_Score, Tumor_Size, MGMT_Methylation, Extent_of_Resection, Radiotherapy, Chemotherapy, Immunotherapy],
                    outputs = [label1]
                )

                y2_predict_btn.click(
                    y2_predict,
                    inputs = [Age, Sex, Race, Hispanic_Ethnicity, Insurance_Status, Facility_Type, Facility_Location, CharlsonDeyo_Score, Tumor_Size, MGMT_Methylation, Extent_of_Resection, Radiotherapy, Chemotherapy, Immunotherapy],
                    outputs = [label2]
                )
                
                y3_predict_btn.click(
                    y3_predict,
                    inputs = [Age, Sex, Race, Hispanic_Ethnicity, Insurance_Status, Facility_Type, Facility_Location, CharlsonDeyo_Score, Tumor_Size, MGMT_Methylation, Extent_of_Resection, Radiotherapy, Chemotherapy, Immunotherapy],
                    outputs = [label3]
                )
                
                y4_predict_btn.click(
                    y4_predict,
                    inputs = [Age, Sex, Race, Hispanic_Ethnicity, Insurance_Status, Facility_Type, Facility_Location, CharlsonDeyo_Score, Tumor_Size, MGMT_Methylation, Extent_of_Resection, Radiotherapy, Chemotherapy, Immunotherapy],
                    outputs = [label4]
                )                

                y1_interpret_btn.click(
                    y1_interpret,
                    inputs = [Age, Sex, Race, Hispanic_Ethnicity, Insurance_Status, Facility_Type, Facility_Location, CharlsonDeyo_Score, Tumor_Size, MGMT_Methylation, Extent_of_Resection, Radiotherapy, Chemotherapy, Immunotherapy],
                    outputs = [plot1],
                )
                
                y2_interpret_btn.click(
                    y2_interpret,
                    inputs = [Age, Sex, Race, Hispanic_Ethnicity, Insurance_Status, Facility_Type, Facility_Location, CharlsonDeyo_Score, Tumor_Size, MGMT_Methylation, Extent_of_Resection, Radiotherapy, Chemotherapy, Immunotherapy],
                    outputs = [plot2],
                )

                y3_interpret_btn.click(
                    y3_interpret,
                    inputs = [Age, Sex, Race, Hispanic_Ethnicity, Insurance_Status, Facility_Type, Facility_Location, CharlsonDeyo_Score, Tumor_Size, MGMT_Methylation, Extent_of_Resection, Radiotherapy, Chemotherapy, Immunotherapy],
                  outputs = [plot3],
                )
                
                y4_interpret_btn.click(
                    y4_interpret,
                    inputs = [Age, Sex, Race, Hispanic_Ethnicity, Insurance_Status, Facility_Type, Facility_Location, CharlsonDeyo_Score, Tumor_Size, MGMT_Methylation, Extent_of_Resection, Radiotherapy, Chemotherapy, Immunotherapy],
                  outputs = [plot4],
                )                
                
    gr.Markdown(
                """    
                <center><h2>Disclaimer</h2>
                <center> 
                The data utilized for this tool is sourced from the Commission on Cancer (CoC) of the American College of Surgeons and the American Cancer Society. These institutions, however, have not verified the information and are not responsible for the statistical validity of the data analysis or the conclusions drawn by the authors. This predictive tool, available on this webpage, is designed to provide general health information only and is not a substitute for professional medical advice, diagnosis, or treatment. It is strongly recommended that users consult with their own healthcare provider for any health-related concerns or issues. The authors make no warranties or representations, express or implied, regarding the accuracy, timeliness, relevance, or utility of the information contained in this tool. The health information in the prediction tool is subject to change and can be affected by various confounders, therefore it may be outdated, incomplete, or incorrect. No doctor-patient relationship is created by using this prediction tool and the authors have not validated its content. The authors do not record any specific user information or initiate contact with users. Before making any healthcare decisions or taking or refraining from any action based on the information in this prediction tool, it is advisable to seek professional advice from a healthcare provider. By using the prediction tool, users acknowledge and agree that neither the authors nor any other party will be liable for any decisions made, actions taken or not taken as a result of the information provided herein.
                <br/>
                <h4>By using this tool, you accept all of the above terms.<h4/>
                </center>
                """
    )                
                
demo.launch()