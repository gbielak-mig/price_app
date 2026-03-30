```python
import streamlit as st
import pandas as pd
import requests
from io import BytesIO, StringIO
from requests.auth import HTTPBasicAuth

# ============================================================
# LOGOWANIE
# ============================================================
APP_PASSWORD = st.secrets["app"]["password"]

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Logowanie")
    password_input = st.text_input("Hasło:", type="password")

    if st.button("Zaloguj"):
        if password_input == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Błędne hasło")

    st.stop()

# ============================================================

st.set_page_config(layout="wide")
st.title("Porównanie MPK")

HTTP_USERNAME = st.secrets["http_auth"]["username"]
HTTP_PASSWORD = st.secrets["http_auth"]["password"]

SHOP_TO_MPK = {
    '50stylepl': 'S501','butysportowe': 'S503','sizeerpl': 'S500','sizeerde': 'G500',
    'sizeercz': 'CZ50','sizeersk': 'SK50','sizeerlt': 'LT50','sizeerro': 'RO50',
    'timberland': 'S502','jdsportspl': 'S512','jdsportsro': 'RO55','jdsportssk': 'SK52',
}

SHOP_DICT = {name: url for name, url in st.secrets["shop_urls"].items()}
MPK_TO_SHOP = {SHOP_TO_MPK.get(shop, shop): shop for shop in SHOP_DICT.keys()}

def get_mpk_code(shop_name):
    return SHOP_TO_MPK.get(shop_name, shop_name)

def load_csv(url):
    resp = requests.get(url, auth=HTTPBasicAuth(HTTP_USERNAME, HTTP_PASSWORD))
    resp.raise_for_status()
    sep = ';' if ';' in resp.text.splitlines()[0] else ','
    return pd.read_csv(StringIO(resp.text), sep=sep)

# ============================================================
# WYBÓR SKLEPÓW
# ============================================================

selected_mpk = st.multiselect("Wybierz MPK", list(MPK_TO_SHOP.keys()))

if len(selected_mpk) != 2:
    st.info("Wybierz dokładnie 2 MPK")
    st.stop()

shop1, shop2 = [MPK_TO_SHOP[x] for x in selected_mpk]
mpk1, mpk2 = selected_mpk

df1 = load_csv(SHOP_DICT[shop1])
df2 = load_csv(SHOP_DICT[shop2])

merged = pd.merge(df1, df2, on="Index", suffixes=(f"_{mpk1}", f"_{mpk2}"))

# ============================================================
# OBLICZENIA
# ============================================================

merged['Price_Diff'] = merged[f'Price_{mpk1}'] - merged[f'Price_{mpk2}']
merged['Price_Diff_%'] = (merged['Price_Diff'] / merged[f'Price_{mpk2}'].replace(0, pd.NA)) * 100

merged['Variants_Diff'] = merged[f'Variants_{mpk1}'] - merged[f'Variants_{mpk2}']
merged['Variants_Diff_%'] = (merged['Variants_Diff'] / merged[f'Variants_{mpk2}'].replace(0, pd.NA)) * 100

merged['Quantity_Diff'] = merged[f'Quantity_{mpk1}'] - merged[f'Quantity_{mpk2}']
merged['Quantity_Diff_%'] = (merged['Quantity_Diff'] / merged[f'Quantity_{mpk2}'].replace(0, pd.NA)) * 100

for col in ['Price_Diff_%','Variants_Diff_%','Quantity_Diff_%']:
    merged[col] = merged[col].round(2)

result = pd.DataFrame({
    'Index': merged['Index'],
    'ID': merged[f'ID_{mpk1}'],
    'Brand': merged[f'Brand_{mpk1}'],

    f'Price_{mpk1}': merged[f'Price_{mpk1}'],
    f'Price_{mpk2}': merged[f'Price_{mpk2}'],
    'Price_Diff': merged['Price_Diff'],
    'Price_Diff_%': merged['Price_Diff_%'],

    f'Variants_{mpk1}': merged[f'Variants_{mpk1}'],
    f'Variants_{mpk2}': merged[f'Variants_{mpk2}'],
    'Variants_Diff': merged['Variants_Diff'],
    'Variants_Diff_%': merged['Variants_Diff_%'],

    f'Quantity_{mpk1}': merged[f'Quantity_{mpk1}'],
    f'Quantity_{mpk2}': merged[f'Quantity_{mpk2}'],
    'Quantity_Diff': merged['Quantity_Diff'],
    'Quantity_Diff_%': merged['Quantity_Diff_%'],
})

# ============================================================
# 🎨 KOLOROWANIE
# ============================================================

def highlight(val):
    if pd.isna(val):
        return ''
    if abs(val) > 10:
        return 'background-color: #ffcccc'
    elif abs(val) > 5:
        return 'background-color: #fff3cd'
    return ''

styled = result.style.applymap(
    highlight,
    subset=['Price_Diff_%','Variants_Diff_%','Quantity_Diff_%']
)

st.dataframe(styled, use_container_width=True, height=600)

# ============================================================
# DOWNLOAD
# ============================================================

csv = result.to_csv(index=False).encode('utf-8')
st.download_button("📥 CSV", csv, "porownanie.csv")

buffer = BytesIO()
with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
    result.to_excel(writer, index=False)
buffer.seek(0)

st.download_button("📥 XLSX", buffer, "porownanie.xlsx")
```
