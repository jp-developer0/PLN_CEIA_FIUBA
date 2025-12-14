<img src="https://github.com/jp-developer0/PLN_CEIA_FIUBA/blob/main/logoFIUBA.jpg" width="500" align="center">

# CEIA - Procesamiento de Lenguaje Natural I

## Descripción
Este repositorio contiene las soluciones a los desafíos de la asignatura **Procesamiento de Lenguaje Natural I**, correspondiente a la Carrera de Especialización en Inteligencia Artificial de la Universidad de Buenos Aires (UBA). El objetivo de la materia es abordar problemas relacionados con el procesamiento y análisis de texto mediante técnicas de inteligencia artificial y aprendizaje automático.

En este repositorio se incluyen los cuatro desafíos propuestos en la asignatura, cada uno enfocado en diferentes aspectos fundamentales del procesamiento de lenguaje natural.

## Autor
- **Juan Pablo González**

## Estructura del Repositorio

### Desafío 1: Vectorización y Clasificación de Texto
- **Archivo**: [solucion_desafio1_jp.ipynb](clase_1/ejercicios/solucion_desafio1_jp.ipynb)
- **Enfoque**: Vectorización de documentos y clasificación de texto con métodos tradicionales de NLP
- **Técnicas**: TF-IDF, Naive Bayes (MultinomialNB y ComplementNB), clasificación por prototipos, similaridad de coseno
- **Dataset**: 20 Newsgroups (11,314 documentos de entrenamiento, 7,532 de prueba)
- **Principales Resultados**:
  - Clasificación por prototipos: F1-score = 0.5050
  - TF-IDF + MultinomialNB (baseline): F1-score = 0.6338
  - TF-IDF + ComplementNB (mejor configuración): F1-score = 0.6960
  - Análisis de similaridad de palabras mediante matriz documento-término transpuesta
- **Conclusiones**:
  - ComplementNB supera consistentemente a MultinomialNB en datasets desbalanceados
  - La configuración óptima de TF-IDF (bigramas, filtrado de términos raros) mejora significativamente el rendimiento
  - La clasificación por prototipos, aunque simple, ofrece una baseline útil para comparación
  - El análisis de similaridad de palabras revela relaciones semánticas capturadas por la matriz TF-IDF

### Desafío 2: Embeddings Personalizados con Word2Vec
- **Archivo**: [solucion_desafio2_jp.ipynb](clase_2/ejercicios/solucion_desafio2_jp.ipynb)
- **Enfoque**: Creación de embeddings personalizados mediante Word2Vec Skip-gram
- **Técnicas**: Word2Vec Skip-gram, análisis de similitud semántica, t-SNE para visualización, clustering semántico
- **Dataset**: Corpus de letras de Bob Dylan (5,213 líneas, vocabulario de 948 palabras únicas)
- **Configuración del Modelo**:
  - Arquitectura: Skip-gram
  - Dimensión de embeddings: 300
  - Ventana de contexto: 5
  - Épocas de entrenamiento: 30
- **Principales Resultados**:
  - Palabras similares a "love": heart, girl, care, feel, life
  - Palabras similares a "time": night, long, day, times, morning
  - Palabras similares a "road": street, highway, main, track
  - Visualización t-SNE reveló 4 clusters principales:
    - Cluster de emociones (love, heart, feel)
    - Cluster temporal (time, night, day)
    - Cluster de viaje (road, street, highway)
    - Cluster de naturaleza (sky, wind, rain)
- **Conclusiones**:
  - Skip-gram captura efectivamente relaciones semánticas en corpus pequeños y especializados
  - El preprocesamiento (normalización, eliminación de stopwords) es crucial para la calidad de los embeddings
  - La visualización t-SNE permite identificar estructuras temáticas en la poesía de Dylan
  - Los embeddings reflejan la naturaleza poética del corpus, agrupando conceptos por campos semánticos

### Desafío 3: Modelos de Lenguaje con LSTMs
- **Archivo**: [solucion_desafio3_jp.ipynb](clase_3/ejercicios/solucion_desafio3_jp.ipynb)
- **Enfoque**: Generación de secuencias de texto mediante redes LSTM con tokenización a nivel de caracteres
- **Técnicas**: LSTM, teacher forcing, perplexity como métrica de evaluación, early stopping
- **Dataset**: Texto literario para modelado de lenguaje
- **Componentes Implementados**:
  - Callback personalizado de perplexity con optimizaciones de memoria
  - Arquitectura LSTM con embeddings y capas densas
  - Sistema de generación de secuencias
  - Monitoreo de pérdida durante entrenamiento
- **Desafíos Resueltos**:
  - Optimización de memoria en callbacks de perplexity
  - Prevención de congelamiento durante entrenamiento
  - Ajuste de hiperparámetros para convergencia estable
- **Conclusiones**:
  - La perplexity es una métrica efectiva para evaluar modelos de lenguaje durante el entrenamiento
  - El teacher forcing acelera el aprendizaje pero puede generar dependencia en datos de entrenamiento
  - La optimización de memoria es crítica para entrenar modelos con secuencias largas
  - Los LSTMs capturan patrones sintácticos y semánticos a nivel de caracteres

### Desafío 4: Traducción Secuencia a Secuencia (Seq2Seq)
- **Archivo**: [solucion_desafio4_jp.ipynb](clase_3/ejercicios/solucion_desafio4_jp.ipynb)
- **Enfoque**: Modelo de traducción inglés-español mediante arquitectura encoder-decoder con PyTorch
- **Técnicas**: Seq2Seq, LSTM bidireccional, embeddings pre-entrenados (GloVe), teacher forcing, one-hot encoding optimizado
- **Dataset**: spa-eng (15,000 pares de traducción español-inglés)
- **Configuración del Experimento**:
  - Longitud de secuencias: 20 tokens (español), 22 tokens (inglés)
  - Embeddings: GloVe 100 dimensiones (congelados)
  - Learning rate: 0.001
  - Arquitecturas evaluadas: LSTM-64, LSTM-128, LSTM-256
- **Optimizaciones Críticas**:
  - Eliminación de softmax en el decoder (se aplica automáticamente en CrossEntropyLoss)
  - One-hot encoding optimizado para evitar problemas de memoria
  - Early stopping para prevenir overfitting
  - Embeddings pre-entrenados congelados
- **Principales Resultados**:
  - Los tres modelos (64, 128, 256 unidades) mostraron rendimiento comparable
  - Ejemplos de traducción del dataset muestran precisión en frases simples
  - Traducciones personalizadas demuestran capacidad de generalización
  - El modelo captura estructuras gramaticales básicas inglés-español
- **Conclusiones**:
  - La arquitectura encoder-decoder es efectiva para traducción con datasets moderados
  - Los embeddings GloVe pre-entrenados aceleran la convergencia y mejoran la calidad
  - El tamaño del hidden state (64-256) tiene impacto limitado; la calidad de datos es más crítica
  - El teacher forcing es esencial para entrenar modelos Seq2Seq efectivos
  - La optimización de memoria (one-hot encoding, manejo de tensores) es crucial en PyTorch

## Conclusiones Generales del Curso

### Progresión de Técnicas de NLP
El curso presenta una progresión natural desde métodos tradicionales hasta arquitecturas de deep learning:
1. **Métodos estadísticos** (TF-IDF, Naive Bayes): Baseline sólida, interpretable y eficiente
2. **Embeddings densos** (Word2Vec): Captura de relaciones semánticas mediante vectores continuos
3. **Modelos secuenciales** (LSTM): Modelado de dependencias temporales en texto
4. **Arquitecturas encoder-decoder** (Seq2Seq): Transformación de secuencias para tareas como traducción

### Aspectos Técnicos Fundamentales

1. **Preprocesamiento**:
   - Crítico para todos los modelos de NLP
   - Impacta directamente en la calidad de vectorización y embeddings
   - Debe adaptarse al dominio y tarea específica

2. **Selección de Arquitecturas**:
   - Para clasificación de texto: ComplementNB con TF-IDF optimizado
   - Para embeddings semánticos: Skip-gram para corpus especializados
   - Para generación de texto: LSTM con teacher forcing
   - Para traducción: Encoder-Decoder con embeddings pre-entrenados

3. **Optimización y Regularización**:
   - Early stopping previene overfitting en modelos recurrentes
   - Embeddings congelados reducen parámetros entrenables
   - Gestión de memoria es crítica en PyTorch (especialmente one-hot encoding)
   - La elección de learning rate impacta significativamente la convergencia

### Metodologías de Evaluación
- **F1-score macro**: Métrica balanceada para clasificación multiclase
- **Perplexity**: Evaluación de modelos de lenguaje durante entrenamiento
- **Análisis cualitativo**: Inspección manual de salidas complementa métricas cuantitativas
- **Métricas específicas de tarea**: BLEU para traducción (recomendado vs. accuracy)

### Desafíos y Limitaciones Identificadas

1. **Trade-off Complejidad vs. Datos**:
   - Modelos complejos (LSTM-256) no superan significativamente a versiones más simples (LSTM-64) con datasets limitados
   - La calidad y cantidad de datos es más importante que la complejidad arquitectural

2. **Overfitting**:
   - Especialmente crítico en modelos Seq2Seq con datasets pequeños
   - Requiere early stopping, dropout, y validación cuidadosa

3. **Optimización de Memoria**:
   - One-hot encoding puede causar explosión de memoria
   - Callbacks de perplexity requieren optimización para datasets grandes

4. **Elección de Métricas**:
   - Accuracy no es adecuada para traducción (usar BLEU, METEOR)
   - F1-score macro preferible a accuracy en clasificación desbalanceada

### Recomendaciones para Futuros Trabajos

1. **Datos**: Priorizar calidad y cantidad de datos sobre complejidad del modelo
2. **Validación**: Implementar conjuntos de validación robustos y monitoreo continuo
3. **Métricas**: Seleccionar métricas específicas para cada tarea de NLP
4. **Iteración**: Comenzar con modelos simples (baseline) y aumentar complejidad gradualmente
5. **Análisis**: Combinar evaluación cuantitativa con inspección cualitativa de resultados
6. **Memoria**: Planificar gestión de memoria desde el inicio, especialmente en PyTorch
7. **Embeddings**: Aprovechar embeddings pre-entrenados cuando sea posible

## Tecnologías Utilizadas
- **Python 3.x**
- **PyTorch** para modelos Seq2Seq (Desafío 4)
- **TensorFlow/Keras** para LSTMs (Desafío 3)
- **scikit-learn** para clasificadores tradicionales y vectorización TF-IDF
- **Gensim** para Word2Vec
- **NumPy** y **Pandas** para manipulación de datos
- **Matplotlib** y **Seaborn** para visualizaciones
- **NLTK** para preprocesamiento de texto

## Cómo Ejecutar los Notebooks

1. **Clonar el repositorio**:
```bash
git clone <repository-url>
cd procesamiento_lenguaje_natural
```

2. **Instalar dependencias**:
```bash
pip install torch torchvision torchaudio
pip install tensorflow keras
pip install scikit-learn gensim nltk
pip install numpy pandas matplotlib seaborn
```

3. **Descargar recursos de NLTK** (si es necesario):
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

4. **Ejecutar notebooks**: Abrir con Jupyter Notebook o JupyterLab
```bash
jupyter notebook
```

## Contacto
Para consultas sobre este proyecto: Juan Pablo González

---
**Universidad de Buenos Aires - Facultad de Ingeniería**\
**Carrera de Especialización en Inteligencia Artificial (CEIA)**\
**Procesamiento de Lenguaje Natural I**
