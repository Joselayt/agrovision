
# **🌿 AgroVision AI — Control de Calidad Agrícola**

Documentación técnica y especificaciones de la arquitectura del modelo de clasificación y control de calidad basado en redes neuronales profundas (PyTorch) y procesamiento de lenguaje natural / características tabulares.

## **📌 1\. Resumen Ejecutivo**

Este módulo forma parte del sistema **AgroVision AI**. Implementa una red neuronal de clasificación construida con **PyTorch** para analizar muestras agrícolas. El pipeline preprocesa datos tabulares e información textual (*feature extraction* mediante TfidfVectorizer), entrena la red clasificadora monitorizando las métricas en tiempo real a través de **Matplotlib**, y exporta tanto los pesos del modelo (.pt) como los transformadores de datos (.gz) para su despliegue en producción.

## **📂 2\. Estructura de Archivos y Artefactos**

Plaintext  
.  
├── datasets/  
│   ├── dataset\_train.csv    \# Conjunto de datos primario de entrenamiento  
│   └── dataset\_test.csv     \# Conjunto de datos primario de evaluación  
├── models/  
│   ├── control-calidad.pt   \# Pesos optimizados del modelo PyTorch  
│   └── vector.gz            \# Modelo TF-IDF serializado con Joblib  
├── training.py              \# Script principal de preprocesamiento y entrenamiento  
└── README.md                \# Documentación técnica del proyecto

## **🛠️ 3\. Preprocesamiento de Datos y Feature Engineering**

El flujo de preprocesamiento prepara y normaliza los datos de entrada antes de ser convertidos a tensores:

> 1. **Fusión de Datasets:** Unificación de dataset\_train.csv y dataset\_test.csv mediante un *vertical stack* (np.vstack) para reprocesar de manera homogénea.  
> 2. **Extracción de Características Textuales:**  
   * La columna nombre es procesada mediante TfidfVectorizer de *Scikit-Learn*.  
   * Se extraen y agregan las componentes matriciales n0 y n1 a la matriz de características.  
> 3. **Limpieza y Selección:**  
   * Eliminación de identificadores directos (nombre) y la variable objetivo (feature / Y).  
   * Asignación del tipo de dato float32 para optimización de memoria.  
> 4. **División del Conjunto de Datos:**  
   * train\_test\_split con una relación $80/20$ ($80\\%$ Entrenamiento, $20\\%$ Test) utilizando random\_state=42 para garantizar la reproducibilidad.  
> 5. **Conversión a Tensores:**  
   * Mapeo de variables de entrada $X$ a torch.float32.  
   * Mapeo de la variable de salida $Y$ a torch.long para clasificación multiclase / binaria.

## **🧠 4\. Arquitectura de la Red Neuronal (classificador)**

El modelo utiliza una red neuronal totalmente conectada (*Fully Connected / Multilayer Perceptron*) diseñada secuencialmente para la extracción de patrones de calidad:

Plaintext  
 \[Entrada: 7 características\]  
             │  
             ▼  
      \[nn.Linear(7, 5)\]  
             │  
             ▼  
        \[nn.ReLU()\]  
             │  
             ▼  
      \[nn.Linear(5, 3)\]  
             │  
             ▼  
        \[nn.ReLU()\]  
             │  
             ▼  
      \[nn.Linear(3, 2)\]  
             │  
             ▼  
  \[Salida: Logits (2 Clases)\]

### **Especificación de Capas**

| Capa | Tipo de Capa | Entrada (Din​) | Salida (Dout​) | Función de Activación |
| :---- | :---- | :---- | :---- | :---- |
| fc1 | nn.Linear | $7$ | $5$ | nn.ReLU() |
| fc2 | nn.Linear | $5$ | $3$ | nn.ReLU() |
| fc3 | nn.Linear | $3$ | $2$ (Features) | *Identidad (Logits)* |

## **⚙️ 5\. Configuración del Entrenamiento**

> * **Función de Pérdida (Loss Function):** nn.CrossEntropyLoss() (Apta para clasificación supervisada).  
> * **Optimizador:** optim.Adam con una tasa de aprendizaje (Learning Rate) de $\\eta \= 10^{-5}$ (0.00001) para una convergencia suave y estable.  
> * **Épocas Totales:** $600,000$ iteraciones.  
> * **Frecuencia de Evaluación:** Cada $10$ épocas se evalúa el rendimiento contra el conjunto de test con model.eval() y accuracy\_score.

## **📊 6\. Monitoreo en Tiempo Real (Matplotlib Dashboard)**

El script cuenta con un sistema de renderizado en vivo interactivo (plt.ion()) que traza dos curvas sincronizadas:

> * **Eje Y Izquierdo (Rojo \- CRITERION):** Evolución de la pérdida (*Cross Entropy Loss*) durante las épocas.  
> * **Eje Y Derecho (Azul \- LEARNING):** Métrica de precisión (*Accuracy*) en el conjunto de validación/test.  
> * **Consola Integrada en Gráfico:** Muestra la duración en tiempo real del entrenamiento, la pérdida actual y el porcentaje de aprendizaje en curso.

## **💾 7\. Exportación y Persistencia**

Al finalizar el bucle de entrenamiento, el script genera automáticamente los artefactos necesarios para la puesta en producción:

> 1. **Modelo PyTorch:** Se guardan las pesas aprendidas (state\_dict) en ./models/control-calidad.pt.  
> 2. **Vectorizador TF-IDF:** Se guarda el transformador de texto entrenado en ./models/vector.gz utilizando joblib.dump para permitir la transformación de nuevas muestras en el backend.

## **🚀 8\. Instrucciones de Ejecución**

### **Prerrequisitos**

Asegúrate de tener instaladas las dependencias necesarias:

Bash  
pip install torch pandas numpy scikit-learn matplotlib joblib

### **Ejecutar el Entrenamiento**

Asegúrate de contar con la carpeta datasets/ y ejecuta:

Bash  
python training.py  
