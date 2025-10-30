from flask import Flask, render_template
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# =======================
# FUNCIONES AUXILIARES
# =======================

def load_data():
    """Carga los datos del CSV remoto desde GitHub"""
    url = "https://raw.githubusercontent.com/JIMENA-eng/Data1/main/data1.csv"
    df = pd.read_csv(url, delimiter=';')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = df.columns.str.strip()
    return df

def calculate_regression(df):
    """Calcula la regresión exponencial para ambos modelos"""
    x = df['Tiempo_segundos'].values
    y = df['Poblacion_bacterias'].values
    ln_y = np.log(y)
    n = len(x)
    sum_x = np.sum(x)
    sum_ln_y = np.sum(ln_y)
    sum_x_ln_y = np.sum(x * ln_y)
    sum_x2 = np.sum(x ** 2)

    b1 = (n * sum_x_ln_y - sum_x * sum_ln_y) / (n * sum_x2 - sum_x ** 2)
    ln_a1 = (sum_ln_y - b1 * sum_x) / n
    a1 = np.exp(ln_a1)
    a2 = a1
    b2 = np.exp(b1)
    return a1, b1, a2, b2

def create_graph(df, a, b, model_type, color):
    """Crea el gráfico de cada modelo"""
    x = df['Tiempo_segundos'].values
    y = df['Poblacion_bacterias'].values
    x_fit = np.linspace(x.min(), x.max(), 300)
    if model_type == 1:
        y_fit = a * np.exp(b * x_fit)
        label = f'Y = {a:.4f}·e^({b:.4f}x)'
    else:
        y_fit = a * (b ** x_fit)
        label = f'Y = {a:.4f}·{b:.4f}^x'

    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, color=color, alpha=0.6, s=50, label='Datos Observados')
    plt.plot(x_fit, y_fit, color=color, linewidth=2.5, label=label)
    plt.xlabel('Tiempo (segundos)')
    plt.ylabel('Población de Bacterias')
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return image_base64

# =======================
# RUTA PRINCIPAL
# =======================

@app.route('/')
def index():
    df = load_data()
    a1, b1, a2, b2 = calculate_regression(df)

    pred_50_m1 = a1 * np.exp(b1 * 50)
    pred_60_m1 = a1 * np.exp(b1 * 60)
    pred_50_m2 = a2 * (b2 ** 50)
    pred_60_m2 = a2 * (b2 ** 60)

    graph1 = create_graph(df, a1, b1, 1, '#7c3aed')
    graph2 = create_graph(df, a2, b2, 2, '#059669')

    return render_template(
        'index.html',
        model1_a=a1, model1_b=b1,
        model2_a=a2, model2_b=b2,
        pred_50_m1=pred_50_m1, pred_60_m1=pred_60_m1,
        pred_50_m2=pred_50_m2, pred_60_m2=pred_60_m2,
        graph1=graph1, graph2=graph2
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)
