import streamlit as st
import pandas as pd
import requests
from io import BytesIO, StringIO
from requests.auth import HTTPBasicAuth
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# LOGOWANIE DO APLIKACJI
# ============================================================
APP_PASSWORD = st.secrets["app"]["password"]

# Sprawdź czy użytkownik jest zalogowany
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Logowanie do aplikacji")
    st.markdown("Wpisz hasło, aby uzyskać dostęp do porównania danych.")
    
    password_input = st.text_input("Hasło:", type="password", key="password_input")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🔓 Zaloguj", use_container_width=True):
            if password_input == APP_PASSWORD:
                st.session_state.authenticated = True
                st.success("✅ Zalogowano pomyślnie!")
                st.rerun()
            else:
                st.error("❌ Nieprawidłowe hasło!")
    
    st.stop()
# ============================================================

st.set_page_config(page_title="Porównanie Price i Variants", layout="wide")

st.set_page_config(page_title="Porównanie Price i Variants", layout="wide")

st.title("Porównanie Price i Variants - Multi-sklepy")

# --- Dane uwierzytelniające z Secrets (bezpieczne!) ---
HTTP_USERNAME = st.secrets["http_auth"]["username"]
HTTP_PASSWORD = st.secrets["http_auth"]["password"]

# --- Słownik tłumaczenia Sklep -> MPK ---
SHOP_TO_MPK = {
    '50stylepl': 'S501',
    'butysportowe': 'S503',
    'sizeerpl': 'S500',
    'sizeerde': 'G500',
    'sizeercz': 'CZ50',
    'sizeersk': 'SK50',
    'sizeerlt': 'LT50',
    'sizeerro': 'RO50',
    'timberland': 'S502',
    'jdsportspl': 'S512',
    'jdsportsro': 'RO55',
    'jdsportssk': 'SK52',
    'jdsportshu': 'HU52',
    'jdsportslt': 'LT52',
    'jdsportsbg': 'BG52',
    'jdsportscz': 'CZ55',
    'jdsportsua': 'UA52',
    'jdsportshr': 'HR52',
    'jdsportssi': 'SI52',
    'jdsportsee': 'EE52',
}

# --- URL-e sklepów z Secrets (bezpieczne!) ---
SHOP_DICT = {name: url for name, url in st.secrets["shop_urls"].items()}
MPK_TO_SHOP = {SHOP_TO_MPK.get(shop, shop): shop for shop in SHOP_DICT.keys()}


def get_mpk_code(shop_name):
    return SHOP_TO_MPK.get(shop_name, shop_name)


def load_csv(url):
    resp = requests.get(url, auth=HTTPBasicAuth(HTTP_USERNAME, HTTP_PASSWORD))
    resp.raise_for_status()
    sep = ';' if ';' in resp.text.splitlines()[0] else ','
    df = pd.read_csv(StringIO(resp.text), sep=sep, on_bad_lines='skip')
    return df


def extract_product_name(url):
    """
    Wyciąga nazwę produktu z URL.
    Jeśli ostatni segment ma mniej niż 6 znaków, bierze przedostatni.
    """
    try:
        url_str = str(url).rstrip('/')
        parts = url_str.split('/')
        
        if len(parts) < 1:
            return ''
        
        last_part = parts[-1]
        
        if len(last_part) < 6 and len(parts) >= 2:
            return parts[-2]
        
        return last_part
    except:
        return ''


# --- Wybór sklepów ---
selected_mpk_codes = st.multiselect(
    "Wybierz sklepy do porównania",
    list(MPK_TO_SHOP.keys()),
    default=[]
)

selected_shops = [MPK_TO_SHOP[mpk] for mpk in selected_mpk_codes]

if len(selected_shops) < 1:
    st.info("👆 Wybierz przynajmniej jeden sklep, aby rozpocząć")
    st.stop()

# --- Wczytanie danych ---
shop_data = {}
for shop_name in selected_shops:
    mpk_code = get_mpk_code(shop_name)
    with st.spinner(f'Wczytuję dane z {mpk_code}...'):
        df = load_csv(SHOP_DICT[shop_name])
        df['ProductName'] = df['URL'].apply(extract_product_name)
        
        for col in ['ID', 'Brand', 'Quantity', 'Variants']:
            if col not in df.columns:
                df[col] = ''
        
        df['Variants'] = pd.to_numeric(df['Variants'], errors='coerce').fillna(0)
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0)
        df['MPK'] = mpk_code
        
        shop_data[shop_name] = df

# --- Porównanie sklepów ---
if len(selected_shops) == 1:
    shop_name = selected_shops[0]
    mpk_code = get_mpk_code(shop_name)
    result_final = shop_data[shop_name][['Index','ID','Brand','Price','Variants','Quantity','MPK']].copy()
    result_final['ProductName'] = shop_data[shop_name]['ProductName']

elif len(selected_shops) == 2:
    shop1, shop2 = selected_shops[0], selected_shops[1]
    mpk1, mpk2 = get_mpk_code(shop1), get_mpk_code(shop2)
    df1, df2 = shop_data[shop1], shop_data[shop2]
    
    merged = pd.merge(df1, df2, on='Index', suffixes=(f'_{mpk1}', f'_{mpk2}'))
    
    if merged.empty:
        st.warning("Brak wspólnych produktów (Index) między wybranymi sklepami")
        st.stop()
    
    merged['Price_Diff'] = merged[f'Price_{mpk1}'] - merged[f'Price_{mpk2}']
    merged['Variants_Diff'] = merged[f'Variants_{mpk1}'] - merged[f'Variants_{mpk2}']
    merged['Quantity_Diff'] = merged[f'Quantity_{mpk1}'] - merged[f'Quantity_{mpk2}']
    
    final_choice = st.radio(
        "Wybierz źródło finalnej wartości", 
        (mpk1, mpk2)
    )
    
    if final_choice == mpk1:
        result_final = merged[[
            'Index',
            f'ID_{mpk1}',
            f'Brand_{mpk1}',
            f'Price_{mpk1}',
            f'Variants_{mpk1}',
            f'Quantity_{mpk1}',
            f'MPK_{mpk1}',
            'Price_Diff',
            'Variants_Diff',
            'Quantity_Diff'
        ]].rename(columns={
            f'ID_{mpk1}': 'ID',
            f'Brand_{mpk1}': 'Brand',
            f'Price_{mpk1}': 'Price',
            f'Variants_{mpk1}': 'Variants',
            f'Quantity_{mpk1}': 'Quantity',
            f'MPK_{mpk1}': 'MPK'
        })
        result_final['ProductName'] = merged[f'ProductName_{mpk1}']
    else:
        result_final = merged[[
            'Index',
            f'ID_{mpk2}',
            f'Brand_{mpk2}',
            f'Price_{mpk2}',
            f'Variants_{mpk2}',
            f'Quantity_{mpk2}',
            f'MPK_{mpk2}',
            'Price_Diff',
            'Variants_Diff',
            'Quantity_Diff'
        ]].rename(columns={
            f'ID_{mpk2}': 'ID',
            f'Brand_{mpk2}': 'Brand',
            f'Price_{mpk2}': 'Price',
            f'Variants_{mpk2}': 'Variants',
            f'Quantity_{mpk2}': 'Quantity',
            f'MPK_{mpk2}': 'MPK'
        })
        result_final['Price_Diff'] = -result_final['Price_Diff']
        result_final['Variants_Diff'] = -result_final['Variants_Diff']
        result_final['Quantity_Diff'] = -result_final['Quantity_Diff']
        result_final['ProductName'] = merged[f'ProductName_{mpk2}']
    
    st.session_state['merged_data'] = merged
    st.session_state['mpk1'] = mpk1
    st.session_state['mpk2'] = mpk2

else:
    st.info("Wybrano więcej niż 2 sklepy - wyświetlam dane dla każdego oddzielnie")
    result_final = pd.DataFrame()
    
    for shop_name in selected_shops:
        mpk_code = get_mpk_code(shop_name)
        df_temp = shop_data[shop_name][['Index','ID','Brand','Price','Variants','Quantity','MPK','ProductName']].copy()
        result_final = pd.concat([result_final, df_temp], ignore_index=True)

# --- Identyfikacja typów kolumn ---
columns_to_skip = ['ProductName', 'Index']
text_columns = []
numeric_columns = []

for col in result_final.columns:
    if col in columns_to_skip:
        continue
    
    if pd.api.types.is_numeric_dtype(result_final[col]):
        numeric_columns.append(col)
    else:
        text_columns.append(col)

# Inicjalizacja session_state dla filtrów
if 'column_filters' not in st.session_state:
    st.session_state['column_filters'] = {}

# --- FILTRY BEZPOŚREDNIO W INTERFEJSIE TABELI ---
st.markdown("---")
st.markdown("### 🔍 Filtry danych")

# Policz aktywne filtry
active_filters_count = 0
for col_name, filter_val in st.session_state['column_filters'].items():
    if col_name in text_columns and filter_val:
        active_filters_count += 1
    elif col_name in numeric_columns and filter_val:
        col_data = result_final[col_name].dropna()
        if len(col_data) > 0:
            min_val = float(col_data.min())
            max_val = float(col_data.max())
            if filter_val != (min_val, max_val):
                active_filters_count += 1

if active_filters_count > 0:
    st.success(f"✅ Aktywnych filtrów: {active_filters_count}")
else:
    st.info("💡 **Jak filtrować:** Kliknij w ekspander z nazwą kolumny (🔽) poniżej, aby ustawić filtry")

# Tworzenie ekspanderów dla każdej kolumny
all_columns = text_columns + numeric_columns

# Wyświetl filtry w siatce
num_cols = 4
for i in range(0, len(all_columns), num_cols):
    cols = st.columns(num_cols)
    
    for idx, col_name in enumerate(all_columns[i:i+num_cols]):
        with cols[idx]:
            with st.expander(f"🔽 {col_name}"):
                if col_name in text_columns:
                    # Filtr tekstowy z wyszukiwaniem
                    unique_vals = sorted(result_final[col_name].dropna().astype(str).unique().tolist())
                    
                    search = st.text_input(
                        "Szukaj",
                        key=f"search_{col_name}",
                        placeholder="Wpisz frazę..."
                    )
                    
                    if search:
                        unique_vals = [v for v in unique_vals if search.lower() in v.lower()]
                    
                    selected = st.multiselect(
                        "Wybierz wartości",
                        options=unique_vals,
                        default=st.session_state['column_filters'].get(col_name, []),
                        key=f"multi_{col_name}"
                    )
                    
                    st.session_state['column_filters'][col_name] = selected
                    
                elif col_name in numeric_columns:
                    # Filtr numeryczny
                    col_data = result_final[col_name].dropna()
                    
                    if len(col_data) > 0:
                        min_val = float(col_data.min())
                        max_val = float(col_data.max())
                        
                        if min_val != max_val:
                            current_range = st.session_state['column_filters'].get(col_name, (min_val, max_val))
                            
                            # Upewnij się, że current_range jest w poprawnym zakresie
                            safe_min = max(min_val, min(current_range[0], max_val))
                            safe_max = min(max_val, max(current_range[1], min_val))
                            
                            range_val = st.slider(
                                "Zakres",
                                min_value=min_val,
                                max_value=max_val,
                                value=(safe_min, safe_max),
                                key=f"slider_{col_name}"
                            )
                            
                            st.session_state['column_filters'][col_name] = range_val
                            
                            # Wyświetl aktualne wartości
                            st.caption(f"Min: {range_val[0]:.2f} | Max: {range_val[1]:.2f}")

# Przycisk resetowania
if st.button("🔄 Resetuj wszystkie filtry", use_container_width=True):
    st.session_state['column_filters'] = {}
    st.rerun()

# Aplikowanie filtrów
filtered_df = result_final.copy()

for col_name, filter_val in st.session_state['column_filters'].items():
    if col_name in text_columns and filter_val:
        filtered_df = filtered_df[filtered_df[col_name].astype(str).isin(filter_val)]
    elif col_name in numeric_columns and filter_val:
        filtered_df = filtered_df[
            (filtered_df[col_name] >= filter_val[0]) & 
            (filtered_df[col_name] <= filter_val[1])
        ]

# --- Wyświetlenie tabeli ---
if filtered_df is not None and not filtered_df.empty:
    st.markdown("---")
    st.subheader(f"Porównanie: {', '.join(selected_mpk_codes)}")
    st.caption(f"Wyświetlono {len(filtered_df)} z {len(result_final)} produktów")
    
    display_columns = [col for col in filtered_df.columns if col != 'ProductName']
    display_df = filtered_df[display_columns].copy()
    
    edited_df = st.data_editor(
        display_df, 
        use_container_width=True, 
        height=500, 
        num_rows="fixed",
        key="data_editor"
    )
    
    edited_df['ProductName'] = filtered_df['ProductName'].values
    
    # --- Przyciski pobierania ---
    st.markdown("---")
    st.markdown("### 📥 Pobierz dane") 
    col1, col2 = st.columns(2)
    
    with col1:
        csv = edited_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Pobierz CSV (przefiltrowane dane)",
            data=csv,
            file_name=f"porownanie_{'_'.join(selected_mpk_codes)}_filtered.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            edited_df.to_excel(writer, index=False, sheet_name='Porównanie')
        buffer.seek(0)
        
        st.download_button(
            label="📥 Pobierz XLSX (przefiltrowane dane)",
            data=buffer,
            file_name=f"porownanie_{'_'.join(selected_mpk_codes)}_filtered.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    # --- PORÓWNANIE WSPÓLNYCH WARTOŚCI MIĘDZY 2 SKLEPAMI ---
    if len(selected_shops) == 2 and 'merged_data' in st.session_state:
        st.markdown("---")
        st.markdown("### 📊 Porównanie wspólnych wartości między sklepami")
        
        merged = st.session_state['merged_data']
        mpk1 = st.session_state['mpk1']
        mpk2 = st.session_state['mpk2']
        
        # Konfiguracja porównania
        col_cfg1, col_cfg2, col_cfg3, col_cfg4 = st.columns(4)
        
        with col_cfg1:
            # Wybór zmiennej do porównania
            comparison_var = st.selectbox(
                "Zmienna do porównania",
                options=text_columns,
                index=0 if 'Brand' in text_columns else 0,
                key="comparison_variable"
            )
        
        with col_cfg2:
            # Liczba top wartości
            top_n_comparison = st.number_input(
                "Liczba top wartości",
                min_value=1,
                max_value=20,
                value=5,
                key="top_n_comparison"
            )
        
        with col_cfg3:
            # Wybór metryki
            comparison_metric = st.selectbox(
                "Metryka",
                options=numeric_columns,
                index=0 if 'Price' in numeric_columns else 0,
                key="comparison_metric"
            )
        
        with col_cfg4:
            # Wybór widoku
            view_mode = st.radio(
                "Widok",
                ["📈 Boxplot", "📋 Tabela"],
                key="view_mode"
            )
        
        # Znajdź wspólne wartości zmiennej
        values_mpk1 = set(merged[f'{comparison_var}_{mpk1}'].dropna().unique())
        values_mpk2 = set(merged[f'{comparison_var}_{mpk2}'].dropna().unique())
        common_values = sorted(list(values_mpk1 & values_mpk2))
        
        if common_values:
            st.caption(f"Znaleziono {len(common_values)} wspólnych wartości dla '{comparison_var}'")
            
            # Przygotuj dane
            comparison_data = []
            for value in common_values:
                # Dane z mpk1
                data_mpk1 = merged[merged[f'{comparison_var}_{mpk1}'] == value]
                for _, row in data_mpk1.iterrows():
                    comparison_data.append({
                        comparison_var: value,
                        'MPK': mpk1,
                        comparison_metric: row[f'{comparison_metric}_{mpk1}']
                    })
                
                # Dane z mpk2
                data_mpk2 = merged[merged[f'{comparison_var}_{mpk2}'] == value]
                for _, row in data_mpk2.iterrows():
                    comparison_data.append({
                        comparison_var: value,
                        'MPK': mpk2,
                        comparison_metric: row[f'{comparison_metric}_{mpk2}']
                    })
            
            comp_df = pd.DataFrame(comparison_data)
            comp_df = comp_df[comp_df[comparison_metric] > 0]
            
            # Weź top N wartości
            top_values_comparison = comp_df[comparison_var].value_counts().head(top_n_comparison).index.tolist()
            comp_df = comp_df[comp_df[comparison_var].isin(top_values_comparison)]
            
            if view_mode == "📈 Boxplot":
                # Wykres boxplot
                fig = px.box(
                    comp_df,
                    x=comparison_var,
                    y=comparison_metric,
                    color='MPK',
                    title=f'Porównanie {comparison_metric} dla top {top_n_comparison} {comparison_var}: {mpk1} vs {mpk2}',
                    labels={comparison_metric: comparison_metric, comparison_var: comparison_var},
                    color_discrete_map={mpk1: '#1f77b4', mpk2: '#ff7f0e'}
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                # Tabela statystyk
                stats_list = []
                for value in top_values_comparison:
                    val_mpk1 = comp_df[(comp_df[comparison_var] == value) & (comp_df['MPK'] == mpk1)][comparison_metric]
                    val_mpk2 = comp_df[(comp_df[comparison_var] == value) & (comp_df['MPK'] == mpk2)][comparison_metric]
                    
                    if len(val_mpk1) > 0 and len(val_mpk2) > 0:
                        stats_list.append({
                            comparison_var: value,
                            f'{mpk1} - Średnia': round(val_mpk1.mean(), 2),
                            f'{mpk1} - Mediana': round(val_mpk1.median(), 2),
                            f'{mpk1} - Liczba': len(val_mpk1),
                            f'{mpk2} - Średnia': round(val_mpk2.mean(), 2),
                            f'{mpk2} - Mediana': round(val_mpk2.median(), 2),
                            f'{mpk2} - Liczba': len(val_mpk2),
                            'Różnica średnich': round(val_mpk1.mean() - val_mpk2.mean(), 2)
                        })
                
                stats_table = pd.DataFrame(stats_list)
                st.dataframe(stats_table, use_container_width=True, height=400)
                
                # Pobierz tabelę
                csv_comp = stats_table.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"📥 Pobierz statystyki (CSV)",
                    data=csv_comp,
                    file_name=f"porownanie_{comparison_var}_{mpk1}_vs_{mpk2}.csv",
                    mime="text/csv"
                )
        else:
            st.info(f"Brak wspólnych wartości '{comparison_var}' między wybranymi sklepami")
    
    # --- ANALIZA WEDŁUG WYBRANEJ ZMIENNEJ ---
    st.markdown("---")
    st.markdown("### 📊 Analiza według wybranej zmiennej")

    col_an1, col_an2, col_an3 = st.columns(3)
    
    with col_an1:
        analysis_var = st.selectbox(
            "Zmienna do analizy",
            options=text_columns,
            index=0 if 'Brand' in text_columns else 0,
            key="analysis_var"
        )
    
    with col_an2:
        top_n_analysis = st.number_input(
            "Liczba top wartości",
            min_value=1,
            max_value=20,
            value=5,
            key="top_n_analysis"
        )
    
    with col_an3:
        analysis_metric = st.selectbox(
            "Metryka",
            options=numeric_columns,
            index=0 if 'Price' in numeric_columns else 0,
            key="analysis_metric"
        )

    if {analysis_metric, analysis_var, 'MPK'}.issubset(edited_df.columns):
        plot_data = edited_df[[analysis_var, analysis_metric, 'MPK']].copy()
        plot_data = plot_data.dropna(subset=[analysis_var, analysis_metric, 'MPK'])
        plot_data = plot_data[plot_data[analysis_metric] > 0]

        if not plot_data.empty:
            # Top N wartości
            top_vals = plot_data[analysis_var].value_counts().head(top_n_analysis).index.tolist()
            plot_data = plot_data[plot_data[analysis_var].isin(top_vals)]

            if not plot_data.empty:
                active_mpks = plot_data['MPK'].unique().tolist()
                num_mpk = len(active_mpks)
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

                if num_mpk == 1:
                    mpk = active_mpks[0]
                    mpk_data = plot_data[plot_data['MPK'] == mpk]

                    fig = go.Figure()

                    for idx, val in enumerate(top_vals):
                        val_data = mpk_data[mpk_data[analysis_var] == val]
                        if not val_data.empty:
                            fig.add_trace(go.Box(
                                y=val_data[analysis_metric].tolist(),
                                name=str(val),
                                marker_color=colors[idx % len(colors)]
                            ))

                    fig.update_layout(
                        title=f'Rozkład {analysis_metric} dla Top {top_n_analysis} {analysis_var} – {mpk}',
                        xaxis_title=analysis_var,
                        yaxis_title=analysis_metric,
                        height=500
                    )

                    st.plotly_chart(fig, use_container_width=True)

                else:
                    fig = make_subplots(
                        rows=1,
                        cols=num_mpk,
                        subplot_titles=active_mpks,
                        horizontal_spacing=0.08
                    )

                    for col_idx, mpk in enumerate(active_mpks, start=1):
                        mpk_data = plot_data[plot_data['MPK'] == mpk]

                        for val_idx, val in enumerate(top_vals):
                            val_data = mpk_data[mpk_data[analysis_var] == val]
                            if not val_data.empty:
                                fig.add_trace(
                                    go.Box(
                                        y=val_data[analysis_metric].tolist(),
                                        name=str(val),
                                        marker_color=colors[val_idx % len(colors)],
                                        showlegend=(col_idx == 1)
                                    ),
                                    row=1,
                                    col=col_idx
                                )

                    fig.update_layout(
                        title=f"Porównanie {analysis_metric} dla Top {top_n_analysis} {analysis_var} według MPK",
                        height=500,
                        showlegend=True
                    )
                    fig.update_yaxes(title_text=analysis_metric)

                    st.plotly_chart(fig, use_container_width=True)

                # Statystyki podstawowe
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"Średnia {analysis_metric}", f"{plot_data[analysis_metric].mean():.2f}")
                with col2:
                    st.metric(f"Mediana {analysis_metric}", f"{plot_data[analysis_metric].median():.2f}")
                with col3:
                    st.metric("Liczba produktów", len(plot_data))

else:
    st.warning("Brak danych po zastosowaniu filtrów. Spróbuj zmienić kryteria filtrowania.")
