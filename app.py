# -------------------------------------------------------------
# 👨‍💻 Developed by: MTMA
# 📧 Email: mtma.1@hotmail.com
# 🌐 Website: iq.sa
#
# تم دمج هذا الكود مع مكتبة Streamlit لتحويله إلى صفحة ويب تفاعلية
# -------------------------------------------------------------

import streamlit as st
import numpy as np

# --- الجزء الأول: إعداد وتدريب النموذج (هذا الجزء لا يتغير) ---

# 1. بيانات التدريب (مساحة، غرف، طوابق، جودة التشطيب، الحي)
x_train = np.array([
    [50, 1, 1, 1, 3], [60, 1, 2, 2, 3], [100, 3, 2, 4, 2],
    [120, 3, 3, 5, 1], [150, 4, 3, 4, 1], [110, 2, 2, 3, 2],
    [170, 4, 4, 5, 1], [200, 5, 4, 5, 1], [90, 2, 2, 2, 2],
    [180, 4, 3, 4, 1],
])
y_train = np.array([380000, 420000, 650000, 780000, 950000, 710000, 1100000, 1350000, 600000, 1200000])

# 3. Feature Scaling
x_mean = np.mean(x_train, axis=0)
x_range = np.max(x_train, axis=0) - np.min(x_train, axis=0)
x_scaled = (x_train - x_mean) / x_range

# 4. إعدادات النموذج
w = np.zeros(x_train.shape[1])
b = 1e-6
alpha = 0.001
epochs = 50000
m = x_train.shape[0]

# 5. دالة الكوست
def compute_cost(x, y, w, b):
    m = x.shape[0]
    y_hat = np.dot(x, w) + b
    cost = (1 / (2 * m)) * np.sum((y_hat - y) ** 2)
    return cost

# 6. تدريب النموذج (سيتم تشغيله مرة واحدة عند بدء التطبيق)
# لا تقلق من الـ print هنا، ستظهر في الطرفية (Terminal) عند التشغيل
# وهي مفيدة للتأكد من أن التدريب قد تم بنجاح
for i in range(epochs):
    y_hat = np.dot(x_scaled, w) + b
    error = y_hat - y_train
    gradient_w = (1 / m) * np.dot(x_scaled.T, error)
    gradient_b = (1 / m) * np.sum(error)
    w = w - alpha * gradient_w
    b = b - alpha * gradient_b
# --- نهاية جزء التدريب ---


# --- الجزء الثاني: بناء واجهة المستخدم التفاعلية باستخدام Streamlit ---

# عنوان الصفحة ووصفها
st.title("BY:MTMA نموذج تقدير أسعار العقارات بالرياض")
st.write("تم تطوير هذا النموذج كأداة تعليمية لفهم أساسيات الانحدار الخطي في تعلم الآلة.")
st.markdown("---") # خط فاصل

# إنشاء أعمدة لتنظيم الواجهة
col1, col2 = st.columns(2)

with col1:
    st.subheader("خصائص العقار")
    # استبدال input() بـ st.number_input لواجهة أفضل
    area = st.number_input("المساحة (متر مربع)", min_value=30, max_value=500, value=120)
    rooms = st.number_input("عدد الغرف", min_value=1, max_value=10, value=3)
    floors = st.number_input("عدد الطوابق", min_value=1, max_value=5, value=2)

with col2:
    st.subheader("تقييمات إضافية")
    # استبدال input() بـ st.slider لواجهة تفاعلية أكثر
    finishing = st.slider("جودة التشطيب (1 سيء - 5 ممتاز)", 1, 5, 4)
    neighborhood = st.slider("مستوى الحي (1 فاخر - 3 شعبي)", 1, 3, 2)

# زر التقدير والعملية التي تحدث عند الضغط عليه
if st.button("تقدير السعر الآن"):
    # 8. تحويل وتطبيق Feature Scaling على بيانات المستخدم
    x_user = np.array([[area, rooms, floors, finishing, neighborhood]])
    x_user_scaled = (x_user - x_mean) / x_range

    # 9. التنبؤ بالسعر
    y_user_pred = np.dot(x_user_scaled, w) + b
    
    # عرض النتيجة النهائية بشكل أنيق على الصفحة
    st.markdown("---")
    st.header("النتيجة")
    st.success(f"💰 السعر التقديري للعقار هو: **{int(y_user_pred[0]):,} ريال سعودي**")
    
    # 10. عرض الملاحظة الهامة
    st.warning("⚠️ النموذج بالأصل يعطيك فكرة عن المفاهيم الرياضية الأساسية بتعلم الآلة وليس بالضرورة يستخدم لقياس أرقام حقيقية.")

# وضع معلومات التواصل في الأسفل
st.markdown("---")
st.write("👨‍💻 Developed by: MTMA | 📧 Email: mtma.1@hotmail.com | 🌐 Website: iq.sa")