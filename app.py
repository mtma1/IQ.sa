# =============================================================
# 👨‍💻 مطور الكود: MTMA
# 📧 البريد: mtma.1@hotmail.com
# 🌐 الموقع: iq.sa
#
# تاريخ آخر تحديث: 6 أغسطس 2025
# =============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# --- 1. الإعدادات والبيانات ---
st.set_page_config(
    page_title="مختبر MTMA", 
    layout="wide", 
    page_icon="🔬", 
    initial_sidebar_state="expanded"
)

# بيانات التدريب ثابتة
x_train_data = np.array([
    [50, 1, 1, 1, 3.0, 15], [60, 1, 2, 2, 3.1, 12], [100, 3, 2, 4, 2.0, 5], [120, 3, 3, 5, 1.0, 2],
    [150, 4, 3, 4, 1.1, 3], [110, 2, 2, 3, 2.1, 8], [170, 4, 4, 5, 1.2, 1], [200, 5, 4, 5, 1.0, 0],
    [90, 2, 2, 2, 2.2, 18], [180, 4, 3, 4, 1.1, 5], [220, 5, 5, 5, 1.0, 1], [250, 6, 5, 5, 1.0, 0],
    [130, 3, 2, 3, 2.0, 7], [90, 2, 1, 2, 3.2, 20], [300, 6, 6, 5, 1.2, 2], [70, 1, 1, 2, 3.3, 25],
    [350, 7, 6, 5, 1.0, 1], [400, 7, 2, 5, 1.1, 4], [85, 2, 1, 3, 5.0, 10], [140, 3, 2, 4, 4.1, 6],
    [210, 4, 3, 4, 2.2, 3], [160, 4, 2, 3, 6.1, 14], [190, 5, 3, 5, 1.0, 0], [280, 5, 4, 4, 2.0, 5],
    [65, 2, 1, 2, 8.2, 22], [320, 6, 5, 5, 1.1, 2], [125, 3, 2, 4, 3.0, 3], [115, 3, 1, 3, 7.1, 16],
    [260, 5, 4, 5, 1.2, 1], [155, 4, 2, 4, 4.3, 4], [95, 2, 2, 3, 5.2, 9], [175, 4, 3, 4, 2.1, 7],
    [230, 5, 4, 5, 1.0, 2], [80, 2, 1, 2, 9.0, 28], [310, 6, 2, 5, 1.1, 3]
])
y_train_data = np.array([
    320000, 380000, 780000, 950000, 1150000, 750000, 1300000, 1550000, 550000, 1300000,
    1600000, 1800000, 880000, 480000, 2000000, 390000, 2300000, 2600000, 580000, 920000,
    1500000, 950000, 1450000, 1850000, 450000, 2100000, 960000, 610000, 1850000, 1100000,
    740000, 1320000, 1700000, 400000, 2200000
])
FEATURE_NAMES = ['Area', 'Rooms', 'Floors', 'Finishing', 'Neighborhood', 'Age', 'Area^2', 'Age^2', 'Finishing^2']
neighborhood_options = { "Al Olaya (العليا)": 1.0, "Al Malqa (الملقا)": 1.1, "Ad Diriyah (الدرعية)": 1.2, "Al Nakhil (النخيل)": 2.0, "As Sahafah (الصحافة)": 2.1, "Hittin (حطين)": 2.2, "Al Narjis (النرجس)": 3.0, "Ar Rabi (الربيع)": 3.1, "Al Ghadir (الغدير)": 3.2, "An Nada (الندى)": 3.3, "Al Muruj (المروج)": 4.0, "Al Wadi (الوادي)": 4.1, "Al Izdihar (الازدهار)": 4.2, "As Sulimaniyah (السليمانية)": 4.3, "Thumamah (الثمامة)": 4.4, "Al Yarmuk (اليرموك)": 5.0, "Ar Rawdah (الروضة)": 5.1, "Ar Rayyan (الريان)": 5.2, "As Salam (السلام)": 5.3, "Al Khaleej (الخليج)": 5.4, "Al Fayha (الفيحاء)": 6.0, "Al Faisaliyah (الفيصلية)": 6.1, "Al Malaz (الملز)": 6.2, "Ash Shifa (الشفا)": 6.3, "An Naseem (النسيم)": 7.0, "Al Aziziyah (العزيزية)": 7.1, "Al Marwah (المروة)": 7.2, "An Nadhim (النظيم)": 7.3, "As Suwaidi (السويدي)": 8.0, "Dirab (ديراب)": 8.1, "Dhahrat Laban (ظهرة لبن)": 8.2, "Namar (نمار)": 8.3, "Al Urayja (العريجاء)": 9.0, "Badr (بدر)": 9.1, "Irqah (عرقة)": 9.2, "Al Futah (الفوطة)": 9.3, "Al Janadriyah (الجنادرية)": 10.0, "An Nadwah (الندوة)": 10.1, "Al Batha (البطحاء)": 10.2 }

# --- 2. نظام اللغتين والترجمة ---
TEXTS = {
    'lang_choice': {'ar': "اللغة", 'en': "Language"},
    'page_choice': {'ar': "اختر الصفحة", 'en': "Select Page"},
    'page_lab': {'ar': "🔬 المختبر التفاعلي", 'en': "🔬 Interactive Lab"},
    'page_analysis': {'ar': "📊 تحليل الأداء", 'en': "📊 Performance Analysis"},
    'title': {'ar': "مختبر MTMA لتسعير العقارات", 'en': "MTMA's Real Estate Pricing Lab"},
    'intro': {'ar': "أداة تفاعلية عشان تستكشف كيف يأثر تغيير إعدادات التدريب على دقة النموذج.", 'en': "An interactive tool to explore the impact of training parameters on an AI model's accuracy."},
    'training_settings_header': {'ar': "⚙️ معايير تدريب المخ الذكي", 'en': "⚙️ Smart Brain Training Parameters"},
    'optimizer_label': {'ar': "🚀 اختر محرك التدريب", 'en': "🚀 Choose Training Optimizer"},
    'optimizer_help': {'ar': "GD العادي هو الأساسي. Adam المطور أذكى وأسرع غالبًا.", 'en': "Standard GD is the basic one. Adam is often smarter and faster."},
    'poly_label': {'ar': "🧠 تفعيل النظرة المتقدمة (Polynomial)", 'en': "🧠 Enable Advanced View (Polynomial)"},
    'poly_help': {'ar': "يسمح للنموذج بفهم العلاقات المعقدة بين البيانات، يزيد الدقة لكن يبطئ التدريب.", 'en': "Allows the model to understand complex data relationships, increasing accuracy but slowing down training."},
    'epochs_label': {'ar': "🔄 كم مرة تبغى المخ يتدرب؟ (Epochs)", 'en': "🔄 How many training cycles for the brain? (Epochs)"},
    'epochs_help': {'ar': "يمثل كم مرة النموذج يراجع الداتا كاملة. كلما زاد، تعلم النموذج أكثر وصار أدق (بس ياخذ وقت أطول).", 'en': "The number of times the model reviews the entire dataset. A higher value means more learning and better accuracy (but takes longer)."},
    'alpha_label': {'ar': "⚡️ سرعة تعلم المخ (Alpha)", 'en': "⚡️ Brain's Learning Rate (Alpha)"},
    'alpha_help': {'ar': "تتحكم في حجم خطوات التصحيح. رقم صغير يعني تعلم بطيء وحذر، ورقم كبير يعني تعلم سريع وممكن 'يشطح' ويتجاوز الحل الصح.", 'en': "Controls the size of corrective steps. A small value means slow, careful learning; a large value might 'overshoot' the correct solution."},
    'property_details_header': {'ar': "🏡 مواصفات العقار اللي تبي تسعّره", 'en': "🏡 Details of The Property to be Priced"},
    'area_label': {'ar': "📐 المساحة (بالمتر المربع)", 'en': "📐 Area (in square meters)"},
    'rooms_label': {'ar': "🛏️ عدد الغرف", 'en': "🛏️ Number of Rooms"},
    'floors_label': {'ar': "🏢 عدد الطوابق", 'en': "🏢 Number of Floors"},
    'finishing_label': {'ar': "✨ جودة التشطيب", 'en': "✨ Finishing Quality"},
    'age_label': {'ar': "⏳ عمر العقار (بالسنوات)", 'en': "⏳ Property Age (in years)"},
    'neighborhood_label': {'ar': "📍 اختر الحي", 'en': "📍 Select Neighborhood"},
    'button_text': {'ar': "حلّل وسعّر!", 'en': "Analyze & Price!"},
    'spinner_text': {'ar': "لحظات... المخ الذكي يفكر ويتدرب", 'en': "Just a moment... the smart brain is thinking and training"},
    'results_header': {'ar': "📊 نتيجة تحليلك وتخمين السعر", 'en': "📊 Your Analysis & Price Estimation"},
    'price_label': {'ar': "💰 السعر التقديري (SAR)", 'en': "💰 Estimated Price (SAR)"},
    'initial_cost_label': {'ar': "📉 الخطأ قبل التدريب (Cost)", 'en': "📉 Initial Cost (Before Training)"},
    'final_cost_label': {'ar': "📉 الخطأ بعد التدريب (Cost)", 'en': "📉 Final Cost (After Training)"},
    'cost_help': {'ar': "هذا الرقم يمثل متوسط مربع الخطأ في النموذج. هدفنا هو تقليله قدر الإمكان.", 'en': "This number represents the Mean Squared Error of the model. Our goal is to minimize it as much as possible."},
    'rmse_label': {'ar': "🎯 متوسط الخطأ (RMSE)", 'en': "🎯 Average Error (RMSE)"},
    'rmse_help': {'ar': "يمثل متوسط الخطأ الفعلي في توقعات النموذج (بالآف الريالات). رقم أقل يعني دقة أعلى.", 'en': "Represents the average actual prediction error in the same unit as the price (K SAR). A lower number means higher accuracy."},
    'tip_header': {'ar': "💡 وش تعني هالأرقام؟ | بقلم MTMA", 'en': "💡 What Do These Numbers Mean? | by MTMA"},
    'tip_text': {'ar': "**- زيادة دورات التدريب (Epochs):** بتلاحظ إن رقم 'الخطأ بعد التدريب' يصغر، وهذا يعني إن النموذج صار أدق. بس لو زدته أكثر من اللازم ممكن النموذج يحفظ الداتا حفظ مو فهم.\n\n**- تعديل سرعة التعلم (Alpha):** لو كان الرقم صغير مرة، بيكون التعلم بطيء. لو كان كبير مرة، بتلاحظ إن 'الخطأ بعد التدريب' يصير رقم فلكي، يعني النموذج 'شطح' وما قدر يتعلم صح.", 'en': "**- Increasing Epochs:** You'll notice the 'Final Cost' number gets smaller, which means the model is getting more accurate. But if you increase it too much, the model might just memorize the data, not learn from it.\n\n**- Adjusting Alpha:** If the rate is too small, learning will be slow. If it's too large, you'll notice the 'Final Cost' becomes a huge number, meaning the model 'overshoot' and failed to learn properly."},
    'divergence_error': {'ar': "🚨 **فشل التدريب (Divergence)!**\n\nقيمة 'الخطأ بعد التدريب' صارت فلكية وهذا يعني أن **سرعة التعلم (Alpha) كانت عالية جدًا** والنموذج 'شطح'. جرب قيمة أصغر.", 'en': "🚨 **Training Failed (Divergence)!**\n\nThe 'Final Cost' became astronomical, which means the **Learning Rate (Alpha) was too high** and the model diverged. Try a smaller value."},
    'analysis_title': {'ar': "تحليل أداء النموذج", 'en': "Model Performance Analysis"},
    'analysis_intro': {'ar': "هنا نشوف كيف أداء النموذج على بيانات التدريب. الرسوم البيانية تقارن بين الأسعار الحقيقية والأسعار اللي توقعها النموذج.", 'en': "Here we see how the model performs on the training data. The charts compare the actual prices with the prices predicted by the model."},
    'run_lab_first': {'ar': "لازم أول شي تسوي تحليل في صفحة 'المختبر التفاعلي' عشان تطلع لك النتائج هنا.", 'en': "You must first run an analysis on the 'Interactive Lab' page to see the results here."},
    'summary_header': {'ar': "ملخص الإعدادات المستخدمة", 'en': "Summary of Parameters Used"},
    'summary_optimizer': {'ar': "محرك التدريب", 'en': "Optimizer"},
    'summary_poly': {'ar': "النظرة المتقدمة", 'en': "Advanced View"},
    'summary_epochs': {'ar': "دورات التدريب", 'en': "Epochs"},
    'summary_alpha': {'ar': "سرعة التعلم", 'en': "Learning Rate"},
    'summary_final_cost': {'ar': "الخطأ النهائي", 'en': "Final Cost"},
    'charts_header': {'ar': "التحليل البياني (باللغة الإنجليزية)", 'en': "Visual Analysis (in English)"},
    'chart_actual_vs_pred_title': {'en': "Actual vs. Predicted Prices"},
    'chart_actual_vs_pred_xaxis': {'en': "Actual Price (in Thousands SAR)"},
    'chart_actual_vs_pred_yaxis': {'en': "Predicted Price (in Thousands SAR)"},
    'chart_residuals_title': {'en': "Residuals Plot (Errors)"},
    'chart_residuals_xaxis': {'en': "Predicted Price (in Thousands SAR)"},
    'chart_residuals_yaxis': {'en': "Residual (Actual - Predicted)"},
    'chart_importance_title': {'en': "Feature Importance (Model Weights)"},
    'chart_convergence_title': {'en': "Cost Convergence Curve"},
    'chart_convergence_xaxis': {'en': "Epochs"},
    'chart_convergence_yaxis': {'en': "Cost (MSE)"},
    'table_header': {'ar': "جدول النتائج بالتفصيل (بالآلاف)", 'en': "Detailed Results Table (in Thousands)"},
    'params_header': {'ar': "الأوزان اللي تعلمها المخ الذكي", 'en': "Learned Model Parameters (Weights)"},
    'data_header': {'ar': "بيانات التدريب اللي استخدمناها", 'en': "Training Data Used"}
}

# --- 3. التدريب المباشر مع التقنيات المتقدمة ---
@st.cache_data
def train_live_model(epochs, alpha, use_poly, optimizer):
    print(f"🚀 Training: epochs={epochs}, alpha={alpha}, poly={use_poly}, optimizer={optimizer}")
    
    x_train = x_train_data.copy()
    feature_names = FEATURE_NAMES[:6]
    if use_poly:
        x_train = np.c_[x_train, x_train[:, 0]**2, x_train[:, 5]**2, x_train[:, 3]**2]
        feature_names = FEATURE_NAMES

    y_train_scaled = y_train_data / 1000.0

    x_mean = np.mean(x_train, axis=0)
    x_std = np.std(x_train, axis=0)
    x_std[x_std == 0] = 1 
    x_scaled = (x_train - x_mean) / x_std
    
    w = np.zeros(x_train.shape[1])
    b = 1e-6
    m = x_train.shape[0]

    v_dw, s_dw = np.zeros(w.shape), np.zeros(w.shape)
    v_db, s_db = 0, 0
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8

    initial_cost = (1 / (2 * m)) * np.sum((np.dot(x_scaled, w) + b - y_train_scaled) ** 2)
    cost_history = []

    for i in range(epochs):
        error = np.dot(x_scaled, w) + b - y_train_scaled
        gradient_w = (1 / m) * np.dot(x_scaled.T, error)
        gradient_b = (1 / m) * np.sum(error)
        
        if optimizer == 'Adam':
            v_dw = beta1 * v_dw + (1 - beta1) * gradient_w
            v_db = beta1 * v_db + (1 - beta1) * gradient_b
            s_dw = beta2 * s_dw + (1 - beta2) * (gradient_w**2)
            s_db = beta2 * s_db + (1 - beta2) * (gradient_b**2)
            w -= alpha * (v_dw / (np.sqrt(s_dw) + epsilon))
            b -= alpha * (v_db / (np.sqrt(s_db) + epsilon))
        else: # Standard GD
            w -= alpha * gradient_w
            b -= alpha * gradient_b
        
        if not np.isfinite(w).all() or not np.isfinite(b):
            print("🛑 Divergence detected! Stopping training.")
            final_cost = np.inf 
            cost_history.append(final_cost)
            break

        if i % 1000 == 0:
            cost = (1 / (2 * m)) * np.sum(error**2)
            cost_history.append(cost)
    
    final_cost = cost_history[-1] if 'final_cost' not in locals() else initial_cost
    print(f"✅ Training complete. Final Cost: {final_cost}")
    
    results = {
        "w": w, "b": b, "x_mean": x_mean, "x_std": x_std,
        "initial_cost": initial_cost, "final_cost": final_cost, "cost_history": cost_history,
        "use_poly": use_poly, "x_train": x_train, "x_scaled": x_scaled,
        "y_train_scaled": y_train_scaled, "feature_names": feature_names,
        "hyperparameters": {'optimizer': optimizer, 'poly_features': use_poly, 'epochs': epochs, 'alpha': alpha}
    }
    return results

# --- 4. تعريف الصفحات كوظائف ---

def page_lab(lang):
    st.title(TEXTS['title'][lang])
    st.text(TEXTS['intro'][lang])

    with st.container(border=True):
        st.subheader(TEXTS['training_settings_header'][lang])
        col1, col2, col3 = st.columns(3)
        with col1:
            optimizer_choice = st.selectbox(TEXTS['optimizer_label'][lang], options=["Adam (المطور)", "GD (العادي)"], help=TEXTS['optimizer_help'][lang], index=0)
        with col2:
            epochs_choice = st.selectbox(TEXTS['epochs_label'][lang], options=[10000, 20000, 30000, 50000], index=3, help=TEXTS['epochs_help'][lang])
        with col3:
            # --- [ تعديل ] --- إضافة خيارات ألفا الجديدة
            alpha_options = [0.001, 0.005, 0.01, 0.1, 0.3, 1.0, 3.0]
            alpha_default_index = 2 # 0.01
            alpha_choice = st.selectbox(TEXTS['alpha_label'][lang], options=alpha_options, index=alpha_default_index, help=TEXTS['alpha_help'][lang])
        poly_choice = st.checkbox(TEXTS['poly_label'][lang], value=True, help=TEXTS['poly_help'][lang])

    with st.container(border=True):
        st.subheader(TEXTS['property_details_header'][lang])
        c1, c2 = st.columns(2)
        with c1:
            area = st.slider(TEXTS['area_label'][lang], 50, 1000, 150)
            rooms = st.slider(TEXTS['rooms_label'][lang], 1, 10, 4)
            floors = st.slider(TEXTS['floors_label'][lang], 1, 5, 2)
        with c2:
            finishing = st.slider(TEXTS['finishing_label'][lang], 1, 5, 4)
            property_age = st.slider(TEXTS['age_label'][lang], 0, 30, 5)
            neighborhood_name = st.selectbox(TEXTS['neighborhood_label'][lang], list(neighborhood_options.keys()))

    if st.button(TEXTS['button_text'][lang], type="primary", use_container_width=True):
        spinner_text = f"{TEXTS['spinner_text'][lang]} (Optimizer: {optimizer_choice}, Epochs: {epochs_choice}, Alpha: {alpha_choice})"
        with st.spinner(spinner_text):
            st.session_state.model_results = train_live_model(epochs_choice, alpha_choice, poly_choice, "Adam" if "Adam" in optimizer_choice else "GD")
        
        results = st.session_state.model_results
        
        if np.isinf(results['final_cost']):
            st.error(TEXTS['divergence_error'][lang], icon="🚨")
        else:
            st.header(TEXTS['results_header'][lang])
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                neighborhood_value = neighborhood_options[neighborhood_name]
                user_features = np.array([area, rooms, floors, finishing, neighborhood_value, property_age])
                if results['use_poly']:
                    user_features = np.append(user_features, [user_features[0]**2, user_features[5]**2, user_features[3]**2])
                
                x_user_scaled = (user_features - results['x_mean']) / results['x_std']
                y_user_pred_scaled = np.dot(x_user_scaled, results['w']) + results['b']
                
                st.metric(label=f"{TEXTS['price_label'][lang]} - {neighborhood_name}", value=f"{y_user_pred_scaled:,.0f}K")

            with res_col2:
                rmse = np.sqrt(results['final_cost'])
                st.metric(label=TEXTS['rmse_label'][lang], value=f"± {rmse:,.0f}K", help=TEXTS['rmse_help'][lang])

            res_col3, res_col4 = st.columns(2)
            with res_col3:
                st.metric(label=TEXTS['initial_cost_label'][lang], value=f"{results['initial_cost']:,.2f}")
            with res_col4:
                st.metric(label=TEXTS['final_cost_label'][lang], value=f"{results['final_cost']:,.2f}", 
                          delta=f"{results['final_cost'] - results['initial_cost']:,.2f}", delta_color="inverse",
                          help=TEXTS['cost_help'][lang])
            
            st.info(TEXTS['tip_header'][lang], icon="💡")
            st.markdown(TEXTS['tip_text'][lang])

def page_analysis(lang):
    st.title(TEXTS['analysis_title'][lang])
    if 'model_results' not in st.session_state:
        st.warning(TEXTS['run_lab_first'][lang], icon="👈")
        return

    results = st.session_state.model_results
    params = results['hyperparameters']
    
    with st.expander(TEXTS['summary_header'][lang], expanded=True):
        st.json({
            TEXTS['summary_optimizer'][lang]: params['optimizer'],
            TEXTS['summary_poly'][lang]: params['poly_features'],
            TEXTS['summary_epochs'][lang]: params['epochs'],
            TEXTS['summary_alpha'][lang]: params['alpha'],
            TEXTS['summary_final_cost'][lang]: f"{results['final_cost']:.2f}" if np.isfinite(results['final_cost']) else "Diverged"
        })

    st.subheader(TEXTS['charts_header'][lang])
    y_pred_scaled = np.dot(results['x_scaled'], results['w']) + results['b']
    residuals = results['y_train_scaled'] - y_pred_scaled
    plt.style.use('dark_background')
    matplotlib.rcParams.update({'font.size': 12, 'text.color': 'white', 'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white'})

    fig_conv, ax_conv = plt.subplots(figsize=(12, 6))
    fig_conv.patch.set_facecolor('#0E1117')
    ax_conv.set_facecolor('#0E1117')
    ax_conv.plot(np.arange(len(results['cost_history'])) * 1000, results['cost_history'], color='#FF4B4B', marker='o', markersize=3, linestyle='-')
    ax_conv.set_title(TEXTS['chart_convergence_title']['en'])
    ax_conv.set_xlabel(TEXTS['chart_convergence_xaxis']['en'])
    ax_conv.set_ylabel(TEXTS['chart_convergence_yaxis']['en'])
    ax_conv.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig_conv)

    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        fig1.patch.set_facecolor('#0E1117')
        ax1.set_facecolor('#0E1117')
        ax1.scatter(results['y_train_scaled'], y_pred_scaled, alpha=0.7, color='#B57BFF', label='Predictions')
        ax1.plot([results['y_train_scaled'].min(), results['y_train_scaled'].max()], [results['y_train_scaled'].min(), results['y_train_scaled'].max()], 'r--', lw=2, label='Perfect Fit')
        ax1.set_title(TEXTS['chart_actual_vs_pred_title']['en'])
        ax1.set_xlabel(TEXTS['chart_actual_vs_pred_xaxis']['en'])
        ax1.set_ylabel(TEXTS['chart_actual_vs_pred_yaxis']['en'])
        ax1.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        fig2.patch.set_facecolor('#0E1117')
        ax2.set_facecolor('#0E1117')
        ax2.scatter(y_pred_scaled, residuals, alpha=0.7, color='#00C49A')
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_title(TEXTS['chart_residuals_title']['en'])
        ax2.set_xlabel(TEXTS['chart_residuals_xaxis']['en'])
        ax2.set_ylabel(TEXTS['chart_residuals_yaxis']['en'])
        ax2.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(10, 7))
    fig3.patch.set_facecolor('#0E1117')
    ax3.set_facecolor('#0E1117')
    weights = pd.Series(results['w'], index=results['feature_names'])
    weights.plot(kind='barh', ax=ax3, color='#FFC65C')
    ax3.set_title(TEXTS['chart_importance_title']['en'])
    st.pyplot(fig3)

    with st.expander(TEXTS['table_header'][lang]):
        df = pd.DataFrame({
            'Actual Price (K)': np.round(results['y_train_scaled'], 1),
            'Predicted Price (K)': np.round(y_pred_scaled, 1),
            'Difference (K)': np.round(residuals, 1)
        })
        st.dataframe(df)
        
    with st.expander(TEXTS['params_header'][lang]):
        st.write({"Weights (w)": results['w'].tolist(), "Bias (b)": results['b']})
        
    with st.expander(TEXTS['data_header'][lang]):
        st.dataframe(pd.DataFrame(x_train_data, columns=FEATURE_NAMES[:6]))

# --- 5. المشغل الرئيسي للتطبيق ---
lang_choice = st.sidebar.radio(TEXTS['lang_choice']['ar'], ["العربية", "English"])
lang = 'ar' if lang_choice == "العربية" else 'en'
page_options = [TEXTS['page_lab'][lang], TEXTS['page_analysis'][lang]]
page = st.sidebar.radio(TEXTS['page_choice'][lang], page_options, label_visibility="hidden")
if page == TEXTS['page_lab'][lang]:
    page_lab(lang)
elif page == TEXTS['page_analysis'][lang]:
    page_analysis(lang)
st.sidebar.divider()
st.sidebar.caption("👨‍💻 Developed by: MTMA | 📧 mtma.1@hotmail.com | 🌐 iq.sa")