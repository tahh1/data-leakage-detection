#!/usr/bin/env python
# coding: utf-8

# In[636]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import xgboost as xgb
import shap
from sklearn.metrics import confusion_matrix, classification_report, auc, roc_curve


# # Tabla de Contenidos
# 1. [Costos Marginales](#Costos-Marginales)
# 2. [Construccion de Variables](#Construccion-de-variables)
# 3. [Visualizacion de Datos](#Visualizacion-de-datos)
# 4. [Base para Modelos](#Base-para-modelos)
# 5. [Prediccion de desviaciones del costo marginal (modelo 1)](#Prediccion-modelo1)
# 6. [Prediccion de desviaciones del costo marginal (modelo 2)](#Prediccion-modelo2)
# 7. [Merge con resultados de clima (modelo 3)](#Prediccion-modelo3)
# 8. [Reflexion](#Reflexion)

#    

# ## 1. Costos Marginales<a id='Costos-Marginales'>

# ### 1.1. Carga de archivos de entrada

# In[637]:


# Lista de archivos de entrada
archivo_cm_real = 'costo_marginal_real.csv'
archivo_cm_programado = 'costo_marginal_programado.csv'
base_prediccion = 'base_para_prediccion.csv'
base_para_prediccion = 'base_para_prediccion.csv'
datos_clima = 'datos_clima.csv'


# In[638]:


# Se leen archivos de entrada y se almacenan como DataFrame
cm_real = pd.read_csv(archivo_cm_real)
cm_programado = pd.read_csv(archivo_cm_programado)


# In[639]:


# Numero de filas por archivo
print("# Filas en archivos:")
print(f'    {archivo_cm_real}:\t\t{cm_real.shape[0]}')
print(f'    {archivo_cm_programado}:\t{cm_programado.shape[0]}')


# In[640]:


# Datos tabla "costo_real"
cm_real.head(5)


# In[641]:


# Datos tabla "costo programado"
cm_programado.head(5)


# #### 25 horas en un dia?

# In[642]:


cm_real[cm_real['hora'] == 25]['fecha'].unique()


# Llama la atencion el valor 25 en el campo hora. Al revisar los datos, vemos que corresponde al dia "2019-04-06" en donde hubo cambio de hora en Chile. Para estandarizar los datos, optamos por transformar todas las fechas / horas a UTC. Notar que realizaremos esta operacion en el dataset final (despues de unir las tablas de costo_real y costo_programado)

#  

# #### Registros duplicados?

# In[643]:


# Revision de registros duplicados
print(f"{cm_real[cm_real.duplicated(subset=['barra_mnemotecnico', 'fecha', 'hora'])].shape[0]} Registros duplicados en: 'archivo_cm_real.csv'")


# In[644]:


# Revision de registros duplicados
duplicados_cm_prog = cm_programado[cm_programado.duplicated(subset=['mnemotecnico_barra', 'fecha', 'hora'])]
print(f"{duplicados_cm_prog.shape[0]} Registros duplicados en: 'archivo_cm_programado.csv'\n")
print('Codigos de barras con duplicados:')
list(duplicados_cm_prog['mnemotecnico_barra'].unique())


# In[645]:


# Ejemplo de registro duplicado
cm_programado[(cm_programado['mnemotecnico_barra'] == 'BA01T002SE036T002') &
              (cm_programado['fecha'] == '2019-01-01') &
              (cm_programado['hora'] == 1)]


# Al revisar el ejemplo anterior, vemos que el costo es diferente para dos registros con la misma tupla (barra,fecha,hora). Para manejar este caso se opto por mantener los registros mas recientes (esto asumiendo que el costo programado se ajusto en algun momento en el tiempo y que probablemente son mas cercanos a lo que se requiere)

# In[646]:


# Eliminamos los duplicados
cm_programado.drop_duplicates(subset=['mnemotecnico_barra', 'fecha', 'hora'], inplace=True, keep='last')


# In[647]:


# Revisamos que efectivamente no existan mas duplicados en la tabla
print(f"{cm_programado[cm_programado.duplicated(subset=['mnemotecnico_barra', 'fecha', 'hora'])].shape[0]} Registros duplicados en: 'archivo_cm_programado.csv'")


#  

# ### 1.2 Merge de costos reales con costos programados

# In[648]:


# Join de tabla "costos reales" y "costos marginales"
costo_marginal = pd.merge(cm_real, cm_programado, left_on=['barra_mnemotecnico', 'fecha', 'hora'], right_on=['mnemotecnico_barra', 'fecha', 'hora'], how='inner')


# In[649]:


costo_marginal.shape


# Al cruzar ambas tablas usando "inner" join, vemos que perdemos registros. Esto se debe a 2 alternativas: o bien, tenemos datos reales que no costo programado o tenemos un costo programado para datos en los que aun no se cuenta con el valor real. Tambien pueden ser ambas opciones. Revisamos que puede estar ocurriendo ...

# #### Registros de costos programados sin datos reales

# In[650]:


# Exploremos los registros que estan en la tabla "costos marginales programado" y no en la tabla "costos reales"
costo_marginal_right_join = pd.merge(cm_real, cm_programado, left_on=['barra_mnemotecnico', 'fecha', 'hora'], right_on=['mnemotecnico_barra', 'fecha', 'hora'], how='right')
costo_marginal_right_join['fecha_mes_año'] = costo_marginal_right_join['fecha'].str[:-3]


# In[651]:


# Obtenemos el listado de las barras / meses para las que tenemos datos programados pero no reales
delta_programado_real = costo_marginal_right_join[costo_marginal_right_join['barra_mnemotecnico'].isna()]
delta_programado_real[['mnemotecnico_barra', 'fecha_mes_año']].drop_duplicates()


# En efecto tenemos 150 diferentes tuplas (barra, mes_año) en las que contamos con un "costo_programado" pero no tenemos su costo real. Ademas, llama la atención ademas el codigo con valor "-".

# #### Registros de datos reales sin costo programado

# In[652]:


# Exploremos los registros que estan en la tabla "costos marginales programado" y no en la tabla "costos reales"
costo_marginal_left_join = pd.merge(cm_real, cm_programado, left_on=['barra_mnemotecnico', 'fecha', 'hora'], right_on=['mnemotecnico_barra', 'fecha', 'hora'], how='left')
costo_marginal_left_join['fecha_mes_año'] = costo_marginal_left_join['fecha'].str[:-3]


# In[653]:


# Obtenemos el listado de las barras / meses para las que tenemos datos programados pero no reales
delta_real_programado = costo_marginal_left_join[costo_marginal_left_join['mnemotecnico_barra'].isna()]
delta_real_programado[['barra_mnemotecnico', 'fecha_mes_año']].drop_duplicates()


# Existen 4717 tuplas (barra, mes_año) en las que existen costos reales sin costos programados.

# Teniendo en cuenta estas observaciones trabajaremos solo con los datos que existen en ambas tablas (inner join).

# ### 1.3 Analisis exploratorio

# In[654]:


# Transformamos la fecha y hora a variable "fecha_hora_UTC" para evitar problemas con la hora "25" 
costo_marginal['fecha_hora_UTC'] = pd.to_datetime(costo_marginal['fecha']).dt.tz_localize('America/Santiago').dt.tz_convert('GMT') + \
                                   pd.to_timedelta(costo_marginal['hora'], unit='h')
costo_marginal.rename(columns={'costo_en_dolares' : 'costo_real_dolares',
                               'costo' : 'costo_programado_dolares',}, inplace=True)


# In[655]:


# Valores unicos por columna en tabla final
costo_marginal.nunique()


# In[656]:


# Valores unicos por columna en tabla "costos reales"
cm_real.nunique()


# Vemos que existen 220 diferentes barras para las que se programa el costo de un total de 1020 barras con datos reales. <b>O sea se programa el costo para un 21.5% de las barras.</b>

#  

# In[657]:


# Revisamos un resumen estadistico de los valores de las variables
costo_marginal.describe()


# In[658]:


# Revisamos la distribucion de la variable "costo en dolares" costo
fig, ax = plt.subplots(1,1, figsize=(15,6))

sns.kdeplot(data=costo_marginal['costo_real_dolares'], shade=True, ax=ax);
sns.kdeplot(data=costo_marginal['costo_programado_dolares'], shade=True, ax=ax);


# De los graficos anteriores, vemos que tanto el costo real como el costo programado tienen una distribucion con una "cola" larga hacia la derecha. Vale la pena revisar mas adelante si estos valores efectivamente corresponden a "outliers" o si bien es parte de lo que se quiere predecir. Ademas llama la atencion que el costo programado tenga un par de valores negativos.
# 

#  

# ## 2. Construccion de variables<a id='Construccion-de-variables'>

# In[659]:


# Construimos las variables de desviacion, desviacion porcentual y desviacion categorica
umbral_pct = 0.15

costo_marginal['desviacion'] = costo_marginal['costo_real_dolares'] - costo_marginal['costo_programado_dolares']
costo_marginal['desviacion_pct'] = (costo_marginal['costo_programado_dolares'] - costo_marginal['costo_real_dolares'])/costo_marginal['costo_programado_dolares']
costo_marginal['desviacion_cat'] =  costo_marginal.apply(lambda x: 1 if abs(x['desviacion_pct']) > umbral_pct else 0, axis=1)


# In[660]:


# Revisamos los valores estadisticos de las variables creadas
costo_marginal[['desviacion', 'desviacion_pct', 'desviacion_cat']].describe()


# De la tabla anterior, vemos que existen valores de "desviacion_pct" con valores NaN o inf. Estos se deben probablemente a valores con "costo_programado_dolares" en 0. Revisemos los datos mas a fondo ...

# ### Desviacion_pct, valores inf o NaN

# In[661]:


# Casos en los que "desviacion_pct" tiene valores Infinitos
costo_marginal[np.isinf(costo_marginal['desviacion_pct'])].head()


# In[662]:


# Casos en los que "desviacion_pct" tiene valores NaN
costo_marginal[np.isnan(costo_marginal['desviacion_pct'])].head()


# Podemos concluir que efectivamente los problemas ocurren cuando el valor "costo_programado_dolares" es 0 o bien cuando ambos costos_programados son 0. A continuacion filtraremos estos datos para evitar problemas en futuros analisis pero antes, revisaremos si existe alguna barra con todos sus costos_reales en 0 para responder una de las futuras preguntas del desafio

# In[663]:


# Agrupamos los datos por barra y costo y buscamos algun valor que tenga su maximo en 0
datos_barra_costos_agrupados = costo_marginal.groupby(by=['barra_mnemotecnico'], as_index=False)['costo_real_dolares'].max().sort_values(by='costo_real_dolares', ascending=True)
datos_barra_costos_agrupados[datos_barra_costos_agrupados['costo_real_dolares'] == 0]


# Vemos que la barra "BA01G049SE001G049" es la unica con todos sus costos reales en dolares en 0. Filtramos este y todos los valores con "costo_real_dolares" en 0 a continuacion

# In[664]:


# Casos en los que "desviacion_pct" tiene valores NaN
registros_inf_NaN = costo_marginal[np.isnan(costo_marginal['desviacion_pct']) | 
                                   np.isinf(costo_marginal['desviacion_pct'])].index
costo_marginal.drop(registros_inf_NaN, inplace=True)


# In[665]:


# Revisamos los valores estadisticos sin los registros con problemas
costo_marginal[['desviacion', 'desviacion_pct', 'desviacion_cat']].describe()


# Ahora los valores hacen sentido, revisemos como se ven graficamente a continuacion

#  

# ## 3. Visualizacion de datos<a id='Visualizacion-de-datos'>

# In[666]:


# Generamos una funcion para graficar un periodo de tiempo. Incluimos una variable "freq" para graficar para cierto periodo de tiempo (D-dia, W-week, M-mes, etc)
def time_plot_costo_barra(codigo_barra, fecha_inicial, fecha_final, figsize=(15,6), freq=None):
    plot_data = costo_marginal[(costo_marginal['barra_mnemotecnico'] == codigo_barra) &
                               (costo_marginal['fecha'] > fecha_inicial) &
                               (costo_marginal['fecha'] < fecha_final)]
    plot_data.set_index('fecha_hora_UTC', inplace=True)
    
    if freq:
        plot_data = plot_data.resample(freq).median()
    
    fig, ax = plt.subplots(figsize=figsize)
    plot_data.plot(y='costo_real_dolares', label='cmg_real', ax=ax, alpha=0.8)
    plot_data.plot(y='costo_programado_dolares', label='cmg_prog', ax=ax, alpha=0.8)
    
    plt.ylabel('Costo [USD/MWh]')
    plt.title('Costo v/s Fecha')
    plt.xticks(rotation=45)


# #### 3.1. Plot de datos por hora

# In[667]:


# Grafico por hora
time_plot_costo_barra('BA01G021SE018G021', '2019-01-01', '2019-06-30')


# #### 3.2. Plot de datos por semana y por dia

# In[668]:


# Grafico por dia
time_plot_costo_barra('BA01G021SE018G021', '2019-01-01', '2019-06-30', freq='D');


# In[669]:


# Grafico por semana
time_plot_costo_barra('BA01G021SE018G021', '2019-01-01', '2019-06-30', freq='W');


# Viendo los graficos anterior, se aprecia que existe una desviacion mas marcada en el periodo comprendiendo entre mediados de Enero y fines de Febrero. Efecto vacaciones?

#  

# ### 4. Base para modelos<a id='Base-para-modelos'>

# #### 4.1. Analisis exploratorio

# In[670]:


# Se lee archivo base para prediccion
base_modelo = pd.read_csv(base_para_prediccion)


# In[671]:


base_modelo.head(4)


# In[672]:


# Descripcion archivo
print(f"# Filas: \t\t{base_modelo.shape[0]}")
print(f"# Columnas: \t\t{base_modelo.shape[1]}")


# In[673]:


base_modelo.describe()


# In[674]:


# Tipos de dato
base_modelo.info()


# In[675]:


base_modelo.nunique()


#  

# <u>Algunas observaciones:</u>
#  - variable 'gen_eolica_total_mwh' y 'gen_geotermica_total_mwh' son siempre nulas, por lo que podemos eliminarlas del dataset
#  - Costo marginal porcentual tiene valores Inf. Probablemente debido a valores de costos reales programados en 0

# #### Preproceso de datos

# In[676]:


# Eliminamos las columnas con datos 100% vacios
base_modelo.drop(columns=['gen_eolica_total_mwh', 'gen_geotermica_total_mwh'], inplace=True)


# In[677]:


# Eliminamos las filas donde el valor de "cmg_desv_pct" es infinito
base_modelo[np.isinf(base_modelo['cmg_desv_pct'])]


# In[678]:


base_modelo.drop(base_modelo[np.isinf(base_modelo['cmg_desv_pct'])].index, inplace=True)


#  

# #### 4.2. Creacion de nuevas variables en base a variable de tiempo

# In[679]:


fecha_hora = pd.to_datetime(base_modelo['fecha'])

base_modelo['año'] = fecha_hora.dt.year
base_modelo['mes'] = fecha_hora.dt.month
base_modelo['semana'] = fecha_hora.dt.weekofyear
base_modelo['dia'] = fecha_hora.dt.day
base_modelo['dia_semana'] = fecha_hora.dt.dayofweek

base_modelo['fin_de_semana'] = 0
base_modelo.loc[base_modelo['dia_semana'].isin([5,6]), 'fin_de_semana'] = 1


#  

# #### 4.3. Funcion para graficar subestacion y variable

# In[680]:


# Funcion para graficar una variable de una subestacion para un periodo de fechas
def time_plot_subestacion_variable(codigo_se, variable, fechas_a_graficar):
    
    fig, ax = plt.subplots(figsize=(15,8))
    plot_data = base_modelo.loc[(base_modelo['nemotecnico_se'] == codigo_se) & 
                                (base_modelo['fecha'].str[:10].isin(fechas_a_graficar)), ['fecha', 'hora', variable]].copy()
    sns.lineplot(x='hora', y=variable, hue='fecha', data=plot_data);
    plt.title(f'{variable} por hora (sub-estacion: {codigo_se})')


# ##### 4.3.1. Curva de generacion solar 

# ###### subestacion: SE005T002 ; fechas: 10, 11, 12, 13 y 14 de enero de 2019

# In[681]:


time_plot_subestacion_variable('SE005T002', 'gen_solar_total_mwh', ['2019-01-10', '2019-01-11' ,'2019-01-12', '2019-01-13', '2019-01-14'])


# ###### subestacion: SE127T005 ; fechas: 10, 11, 12, 13 y 14 de enero de 2019

# In[682]:


time_plot_subestacion_variable('SE127T005', 'gen_solar_total_mwh', ['2019-01-10', '2019-01-11' ,'2019-01-12', '2019-01-13', '2019-01-14'])


# En base a los graficos anteriores, es claro que la generacion de energia solar ocurre principalmente entre las 8 AM y 22 PM. Tambien da la impresion que la generacion de energia solar para la segunda subestacion tiene un poco mas ruido o fue menos constante. Efectos del clima? Tal vez estuvo con lluvia o nublado o derechamente podria ser que la subestacion se encuentra en un lugar donde el sol llega de manera menos directa ...

#  

# ##### 4.3.2. Curva de generacion termica

# ###### subestacion: SE020G213 ; fechas: 14, 15, 16, 17 mayo de 2019

# In[683]:


time_plot_subestacion_variable('SE020G213', 'gen_termica_total_mwh', ['2019-05-14', '2019-05-15' ,'2019-05-16', '2019-05-17'])


# ###### subestacion: SE106G216 ; fechas: 14, 15, 16, 17 mayo de 2019

# In[684]:


time_plot_subestacion_variable('SE106G216', 'gen_termica_total_mwh', ['2019-05-14', '2019-05-15' ,'2019-05-16', '2019-05-17'])


# En base a los graficos anteriores, es claro que la energia termica es mas constante que la energia solar respecto al tiempo. Ahora bien, la energia promedio producida en la subestacion "SE020G213" en las mismas fechas fue de 520 [MWh] versus la segunda que produjo 17.2 [MWh]. La diferencia es considerable por lo que habria que revisar si esto de debe a un tema de ubicacion o derechamente algun funcionamiento errado

#  

# ### 5. Prediccion de desviaciones del costo marginal: modelo 1<a id='Prediccion-modelo1'>

# In[685]:


# Creacion variable target. Adaptamos la variable usando de un periodo para predecir la proxima hora
umbral_pct = 15

base_modelo['cmg_desv_ind'] = base_modelo.apply(lambda x: 0 if abs(x['cmg_desv_pct']) <= umbral_pct else 1, axis = 1)
base_modelo.sort_values(by=['nemotecnico_se', 'fecha', 'hora'], inplace=True)
base_modelo['target'] = base_modelo.groupby(by=['nemotecnico_se'])['cmg_desv_ind'].shift(-1)
base_modelo.dropna(subset=["target"], inplace=True)


# In[686]:


# Creacion de features
base_modelo['en_total_mwh'] = base_modelo['gen_hidraulica_total_mwh'].fillna(0) + \
                              base_modelo['gen_solar_total_mwh'].fillna(0) + \
                              base_modelo['gen_termica_total_mwh'].fillna(0)

# Añadimos variables correspondientes a la variable "target" desfasada en el tiempo
periodos_lag = 24

for i in range(1, periodos_lag + 1):
    base_modelo[f"cmg_desv_ind_{i}hrs"] = base_modelo.groupby(by=['nemotecnico_se'])['cmg_desv_ind'].shift(i)


base_modelo.dropna(subset=[f"cmg_desv_ind_{i}hrs" for i in range(1, periodos_lag + 1)], inplace=True)


# In[687]:


# Revisamos que tan desbalanceada esta nuestra variable a predecir
ax = sns.barplot(x="cmg_desv_ind", y="cmg_desv_ind", data=base_modelo, estimator=lambda x: len(x) / len(y) * 100).set_title(f'Proporcion de datos con Target = 1 [{100*sum(base_modelo["cmg_desv_ind"])/len(y):.2f}%]')


# Vemos que el dataset esta desbalanceado 25% vs 75%. Esto es esperable a menos que nuestros datos programados esten muy desviados respecto al real

# In[688]:


# Revisamos la distribucion de la variable "target" versus otras variables categoricas
variables_categoricas = ['hora', 'mes', 'semana', 'dia', 'dia_semana', 'fin_de_semana', 'n_barras']
fig, axes = plt.subplots(2, 4, figsize=(25, 10))
for col, ax in zip(variables_categoricas, axes.flatten()):
    g = sns.barplot(x=col, y="target", data=base_modelo, ax=ax) 


# <u>Algunas impresiones:</u>
#  - Al parecer, es mas probable que los desvios ocurran entre 16:00 y 22:00 hrs (tarde / noche)
#  - Hay pocos desvios de valores reales v/s programados para el mes de Junio
#  - Al parecer, es menos probable que ocurra un desvio los fines de semana
#  - Es mas probable que ocurra un desvio cuando hay mas barras a excepcion del caso de 5 barras. Me parece relevante revisar que pasa en ese caso especifico. Outlier?
#  

# In[689]:


# Revisamos la distribucion de la variable "target" versus otras variables categoricas
variables_continuas = ['gen_hidraulica_total_mwh','gen_solar_total_mwh',
                       'gen_termica_total_mwh', 'en_total_mwh',
                       'demanda_mwh', 'cap_inst_mw']
base_modelo_transformada = pd.melt(base_modelo, 'target', variables_continuas).fillna(0)

g = sns.FacetGrid(base_modelo_transformada, col="variable", hue="target", col_wrap=3, sharex=False, sharey=False,height=5,aspect=1.25)
g.map(sns.kdeplot, "value", shade=True)
g.add_legend();


# No se ve una gran diferencia cuando existe o no una desviacion en las variables continuas. No creo que ayuden mucho para nuestro modelo.

#  

# In[690]:


variables_lag = ["cmg_desv_ind"] + [f"cmg_desv_ind_{i}hrs" for i in range(1, periodos_lag + 1)] 
fig, ax = plt.subplots(1,1, figsize=(20,16))
sns.heatmap(base_modelo[['target'] + variables_lag].corr(), annot=True);


# De los graficos anteriores vemos que el flag de desvio de costo marginal (cmg_desv_ind) junto con el lag de esta misma variable para las horas 1, 2, 3 y 4 parecieran ayudar a predecir nuestra variable "target"

# #### 5.1. Construccion del modelo usando XGBoost

# In[691]:


# Creamos el modelo usando una "semilla" fija para obtener resultados deterministicos. Incluimos tambien un argumento para considerar las clases desbalanceadas
seed = 1234
model = xgb.XGBClassifier(n_estimators=500, learning_rate=0.01, scale_pos_weight=3, seed=seed)


# In[692]:


# Funcion para hacer una division de train/test respetando la estructura de series de tiempo
def timeseries_train_test_split(X, y, test_size):
    """
        Funcion para hacer un "split" de train-test manteniendo la estructura de time-series
    """
    
    test_index = int(len(X)*(1-test_size))
    
    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]
    
    return X_train, X_test, y_train, y_test


# In[693]:


# Transformamos la base modelo en train/test manteniendo la estructura de la serie de tiempo
X = base_modelo.drop(columns='target')
X = X[variables_continuas + variables_categoricas + variables_lag]
y = base_modelo['target']

X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.2)


# In[694]:


# Entrenamos el modelo usando XGBoost
result = model.fit(X_train, y_train)


# In[695]:


# Revisamos los resultados v/s nuestro test set
y_pred = result.predict(X_test)
print((classification_report(y_test, y_pred)))


# In[696]:


# Estimamos el indice AUC (Area under the curve), como metrica para evaluar el modelo
fpr, tpr, _ = roc_curve(y_test, y_pred)
print(f'AUC score: {auc(fpr, tpr)}')


# En termino de resultados, vamos a asumir que los falsos positivos y negativos tienen la misma relevancia por lo que usaremos el F1-score y el AUC como varas para medir la efectividad del modelo. En base a estas metricas vemos que se comporta relativamente bien con un F1-score promedio de 0.82 y un AUC de 0.83

# In[697]:


# Graficamos la importancia de las variables usando la libreria SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")


# Vemos que la variable mas importante del modelo es la misma variable target pero desfasada en una hora. Podemos interpretar esto como que si existe una desviacion en la hora actual , es muy probable que exista una desviacion en la hora siguiente.

#  

#   

# ### 6. Prediccion de desviaciones del costo marginal: modelo 2<a id='Prediccion-modelo2'>

# Si los datos son enviados desde produccion cada 12 horas, debiesemos adaptar el modelo para predecir si existira una desviacion en 12 horas mas. La idea es que con los datos enviados desde produccion podamos generar predicciones de desvio continuamente hasta que nos lleguen los nuevos datos de produccion. En el peor caso tendriamos que generar una prediccion valida con datos de 11 horas atras. Esto obliga a generar una prediccion para 12 horas en el futuro.
# <br>
# <br>
# Algunas ideas a explorar para abordar esta problematica (no creo que alcance a explorarlas todas):
# -  Armar solo un modelo de clasificacion para predecir si habra una desviacion para 12 horas en el futuro
# -  Armar 12 modelos de clasificacion (1 por cada hora) y usarlos para predecir las 12 horas hacia el futuro
# -  Armar solo un modelo de clasificacion que prediga si es que existira una desviacion en algun punto en las proximas 12 horas.
# -  Armar dos modelos de regresion lineal para intentar extrapolar tanto la variable costo real como la variable costo programado y usar estos modelos ya sea como features o como una manera de predecir la desviacion

# In[698]:


# Idea  1 : armar un solo modelo de clasificacion para 12 horas al futuro
base_modelo2 = base_modelo.copy()


# In[699]:


# Desviamos la variable target en 12 horas
desvio_en_hrs = 12
base_modelo2['target'] = base_modelo2.groupby(by=['nemotecnico_se'])['cmg_desv_ind'].shift(-desvio_en_hrs)
base_modelo2.dropna(subset=['target'], inplace=True)


# In[700]:


# Añadimos variables correspondientes a la variable "target" desfasada en el tiempo. Notar que este lag debe hacerse agrupando por barra, de otro modo se mezclaran datos entre barras lo que podria ocasionar errores
periodos_lag = 23

for i in range(1, periodos_lag + 1):
    base_modelo2[f"cmg_desv_ind_{i}hrs"] = base_modelo2.groupby(by=['nemotecnico_se'])['cmg_desv_ind'].shift(i)
base_modelo2.dropna(subset=[f"cmg_desv_ind_{i}hrs" for i in range(1, periodos_lag + 1)], inplace=True)


# In[701]:


base_modelo2.head(3)


# In[702]:


variables_lag = ["cmg_desv_ind"] + [f"cmg_desv_ind_{i}hrs" for i in range(1, periodos_lag + 1)] 
fig, ax = plt.subplots(1,1, figsize=(20,16))
sns.heatmap(base_modelo2[['target'] + variables_lag].corr(), annot=True);


# Vemos que al parecer existe una correlacion de la variable target con ella misma (autocorrelacion) desfasada 12, 13 y 14 horas. Podemos interpretar esto como una periodicidad diaria del proceso, donde si mi "costo programado" estuvo desviado del "costo real" el dia anterior, es mas probable que se desvie tambien el dia siguiente a la misma hora.

# In[703]:


# Construimos el modelo usando XGBoost
model_12hrs = xgb.XGBClassifier(n_estimators=500, learning_rate=0.01, scale_pos_weight=3, seed=seed)


# In[704]:


# Transformamos la base modelo en train/test manteniendo la estructura de la serie de tiempo
X = base_modelo2.drop(columns='target')
X = X[variables_continuas + variables_categoricas + variables_lag]
y = base_modelo2['target']

X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.2)


# In[705]:


# Entrenamos el modelo usando XGBoost
result_12hrs = model_12hrs.fit(X_train, y_train)


# In[706]:


y_pred = result_12hrs.predict(X_test)
print((classification_report(y_test, y_pred)))


# In[707]:


# Estimamos el indice AUC (Area under the curve), como metrica para evaluar el modelo
fpr, tpr, _ = roc_curve(y_test, y_pred)
print(f'AUC score: {auc(fpr, tpr)}')


# Revisando los resultados de nuestro modelo 2 v/s el modelo 1, vemos que efectivamente el performance se ve afectando considerablemente (F1-score:  0.65;  AUC: 0.68). Esto tiene mucho sentido dada la importancia de la variable desfasada en una hora para la prediccion. 

# In[708]:


# Graficamos la importancia de las variables usando la libreria SHAP
explainer = shap.TreeExplainer(model_12hrs)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")


# Tambien vemos que la variable mas importante para la prediccion es la misma variable target pero desfasada en 12 horas lo que reafirma la posibilidad de una periodicidad diaria. Las siguientes variables tambien apuntan a posible estacionalidad por semana, mes, dia y hora.  

#  

# ### 7. Merge con resultados de clima: modelo 3<a Id=Prediccion-modelo3>

# In[709]:


# Leemos los datos de clima
clima = pd.read_csv(datos_clima)
clima.head()


# Revisando los datos del clima, nos damos cuenta que se encuentra abierto por subestacion y fecha. Debido a que no podemos saber con antelacion cual sera la temperatura a cierta hora en cada subestacion, no podemos usar estos datos directamente o estariamos generando "data leakage". Para manejar esto, podemos desfasar los datos en un dia de modo de usar la temperatura de ayer para predecir el target de 12 horas mas.

# In[710]:


# Estimamos la fecha desfasada en un dia
clima.sort_values(by=['subestacion','fecha'], inplace=True)
clima['fecha'] = clima.groupby(by=['subestacion'])['fecha'].shift(-1)


# In[711]:


# Unimos los datos de clima con los datos de la tabla "base_modelo2"
base_modelo3 = pd.merge(base_modelo2, clima, left_on=['nemotecnico_se', 'fecha'], right_on=['subestacion', 'fecha'], right_index=False)


# In[712]:


# Revisamos la distribucion de las variables de clima versus la variable target
variables_clima = ['ALLSKY_SFC_SW_DWN', 'KT', 'PRECTOT', 'RH2M', 'T2M', 'T2MDEW', 'T2M_MAX', 'T2M_MIN', 'TQV', 'TS', 'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS50M', 'WS50M_MAX', 'WS50M_MIN']

base_modelo_transformada = pd.melt(base_modelo3, 'target', variables_clima).fillna(0)
g = sns.FacetGrid(base_modelo_transformada, col="variable", hue="target", col_wrap=4, sharex=False, sharey=False,height=5,aspect=1.25)
g.map(sns.kdeplot, "value", shade=True)
g.add_legend();


# Vemos que no existe una gran diferencia de distribucion entre los datos con o sin desvio entre el "costo real" y el "costo programado". A raiz de lo anterior, no esperamos que haya una gran diferencia en el performance del modelo considerando o no estos datos.

#  

# In[713]:


# Entrenamos un nuevo modelo considerando los datos del clima
model_clima = xgb.XGBClassifier(n_estimators=500, learning_rate=0.01, scale_pos_weight=3, seed=seed)


# In[714]:


# Transformamos la base modelo en train/test manteniendo la estructura de la serie de tiempo
X = base_modelo3.drop(columns='target')
X = X[variables_continuas + variables_categoricas + variables_lag + variables_clima]
y = base_modelo3['target']

X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.2)


# In[715]:


# Entrenamos el modelo usando XGBoost
result_clima = model_clima.fit(X_train, y_train)


# In[716]:


y_pred = result_clima.predict(X_test)
print((classification_report(y_test, y_pred)))


# In[717]:


# Estimamos el indice AUC (Area under the curve), como metrica para evaluar el modelo
fpr, tpr, _ = roc_curve(y_test, y_pred)
print(f'AUC score: {auc(fpr, tpr)}')


# Revisando los resultadoss (F1-score: 0.64  ; AUC: 0.675) , vemos que el modelo se comporta mas menos igual que el modelo 2. Esto era esperable considerando como las variables de clima se distribuian respecto a la variable target.

#  

# In[718]:


# Graficamos la importancia de las variables usando la libreria SHAP
explainer = shap.TreeExplainer(model_clima)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")


# Al revisar la relevancia de las variables para el modelo predictivo, vemos que las variables de clima no aparecen en el top 5, por lo que podemos concluir que el clima no es un gran factor que afecte en el desvio entre el costo real y el costo programado

#  

# ### 8. Reflexion<a Id=Reflexion>

# 1. Contar con un modelo que permita anticiparse a desvios de precios de la energia, permite tomar acciones para mitigar los impactos de este incremento. Algunos ejemplos podrian ser los siguientes:
#     - Ajustar el precio para consumidores finales, traspasando el incremento del costo al precio final de la energia
#     - Transparentar esta informacion con potenciales actores en el mercado de generacion electrica. La idea seria motivar una mayor generacion de electricidad en esos "peak" y de esta forma generar una reduccion en los costos marginales para estos periodos
# 
# 2. Como potenciales casos de uso para este modelo, imagino los siguientes:
#     - Una web app que permita predecir el costo final real de electricidad de una casa utilizando su boleta digital. El modelo actual podria ayudar a ajustar el precio final considerando los desvios en los costos
#     - Distribuir esta informacion con potenciales actores de generacion electrica para motivar el incremento de generacion de electricidad para estos periodos

# In[ ]:




