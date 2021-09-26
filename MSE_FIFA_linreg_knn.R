# PRACTICA FIFA20 - Manuel Saravia Enrech 
# El dataset elegido nos da informacion sobre los futbolistas que aparecen en el videojuego FIFA 20.
# Incluye variables numericas como caracteristicas o sueldo, y variables categoricas como el club o la nacionalidad.
# El dataset contiene mas de 18 mil registros (futbolistas) con mas de 100 variables cada uno.

# PARTE 1: REGRESION LINEAL MULTIPLE
# El objetivo es hacer predicciones sobre la variable 'value_eur' (precio del jugador) usando un subconjunto de variables 
# significativas del dataset.
# Considero que es una propuesta interesante para el mundo de los datos aplicado al negocio del deporte profesional. 
# Poder estimar el precio de mercado de un jugador o bien al menos poder categorizar su rango de precio es muy relevante
# para la actividad de compraventa de jugadores, no solo para clubes de futbol sino que tambien en muchos casos son objeto 
# de inversion por parte de fondos que adquieren los derechos (o un porcentaje) de jugadores.

# PASO 1: Carga de Datos. Dataset 'players_20.csv'
df <- read.csv2('players_20.csv', header=TRUE, sep=',', stringsAsFactors=FALSE)
head(df, n=1)  # 104 variables a mostrar...

# PASO 2: Exploracion y Preparacion de los datos
dim(df)
str(df)
# inspeccionamos la variable target 'value_eur' (precio del jugador) en millones de euros
summary(df$value_eur/1000000)
hist(df$value_eur/1000000)

# La mayoria de las variables no son relevantes para la estimacion del precio de mercado de un futbolista.
# Consideraremos inicialmente las siguientes variables:
# - value_eur (precio del jugador en euros) -> variable target
# - age  (edad)
# - overall (valoracion global del jugador 1-100)
# - potential (valoracion global que se cree que un jugador podria alcanzar 1-100)
# - preferred_foot (diestro o zurdo)
# - international_reputation (valor de 1-5)
# - team_position (29 valores posibles)
# - wage_eur (salario del jugador en euros)
#
# Generamos un nuevo dataset df_fifa con solo estas variables y 
# - cambiaremos la variable preferred_foot a factor
# - cambiaremos la variable team_position a factor
# - Cambiaremos la variable value_eur a millones de euros (Meur)
# - Cambiaremos la wage_eur a miles de euros (Keur)
# NOTA: no usaremos la variable release_clause_eur porque tiene una correlación total con el target value_err 
# y el modelo seria demasiado simple. 

# Preparacion de los datos
vars_selected = c('short_name', 'value_eur', 'overall', 'age','potential', 'preferred_foot', 'international_reputation', 'team_position', 'wage_eur')
df_fifa = df[vars_selected]
df_fifa['preferred_foot'] <- as.factor(df_fifa $preferred_foot)
df_fifa['team_position'] <- as.factor(df_fifa $team_position)
df_fifa['value_eur'] = df_fifa['value_eur'] / 1000000
df_fifa['wage_eur'] = df_fifa['wage_eur'] / 1000
head(df_fifa)
str(df_fifa)
summary(df_fifa)
# Se observa que la variable team_position presenta valores nulos ""
dim(df_fifa[which(df_fifa$team_position != ""),]) # 240 registros con valores nulos
# No consideramos estos registros. Trabajaremos con un conjunto de datos limpio de NA's
df_fifa <- df_fifa[which(df_fifa$team_position != ""),]
dim(df_fifa)

# Correlacion
# Graficos iniciales de dispersion de la variable value_eur (Meur), que es la variable target, con respecto 
# a las variables overall, potential y wage_eur.
plot(df_fifa$wage_eur, df_fifa$value_eur)
# Se ve la correlacion positiva existente.
plot(df_fifa$team_position, df_fifa$value_eur)
plot(df_fifa$overall, df_fifa$value_eur)
plot(df_fifa$potential, df_fifa$value_eur)
# Se ve que valores de overall o potential menores que 70 no tienen practicamente impacto en el precio del jugador, que sera muy bajo. 
# Se ve la correlacion positiva existente.

# Vamos a centrarnos en los jugadores cuyo overall y potential sea mayor o igual a 70. Son los jugadores 
# que nos interesan mas con vistas a posible traspasos. Quedan mas de 5.500 registros.
df_fifa <- df_fifa[which(df_fifa$overall >=70 & df_fifa$potential >=70),]
dim(df_fifa)


# matriz de correlacion: usamos todas variables numericas (int, num)
corvar <- c('value_eur', 'overall', 'age','potential', 'international_reputation', 'wage_eur')
cor(df_fifa[corvar])
# Se muestra la correlacion entre las distintas variables, y fijandonos en la variable target value_eur se observa 
# una correlacion alta con con overall, wage_eur y potential, media-alta con international_reputation e inferior con age. 
# Se observa como la correlacion con age es negativa (a partir de una edad cuanto mayor sea un jugador su precio descendera)

# Graficos de dispersion de value_eur vs resto de variables
pairs(df_fifa[c('value_eur', 'overall', 'age', 'potential', 'international_reputation', 'wage_eur')])
# combinamos correlaciones y dispersiones en una gran tabla de graficas
library(psych)
pairs.panels(df_fifa[c('value_eur', 'overall', 'age', 'potential', 'international_reputation', 'wage_eur')])


# PASO 3: Entrenar el modelo con los datos
# Cogemos 70% datos para train y 30% para test
set.seed(100) # fijo una semilla para poder reproducir
n <- nrow(df_fifa)
ntrain <- round(n*0.70)
datos_indices <- sample(1:n, size=ntrain)
data_train <- df_fifa[datos_indices, -1] # sin la variable short_name
dim(data_train)
data_test <- df_fifa[-datos_indices, -1] # sin la variable short_name
dim(data_test)

modelo <- lm(formula = value_eur ~  ., data = data_train)

# PASO 4: evaluar el rendimiento del modelo
summary(modelo)
# R-squared = 0.8542
# Un 85.4% de la variacion de la variable dependiente (value_err) es explicada por el modelo
# El 50% de los errores estan dentro de los valores del primer y tercer cuartil, por lo que la  mayoria de las predicciones 
# estaban entre 1.43 Meur por encima del valor real y 1.26 Meur por debajo.

# Sobre el poder de prediccion de cada variable (*** indican que son predictivas) destacan: overall, age, wage_eur 
# y algunos valores de team_position.  
# Ya con Pr(>|t|) > 0.05 sigue la variable potential, y más alejadas preferred_foot, international_reputation 
# y la mayoria de valores de team_position no se consideran predictivas. 

# Aplicamos la prediccion al data_test. (Ver modelo2)
Ypred = predict(modelo, data_test)
# Muestra de la prediccion (columna value_eur-pred). Mostramos 5 casos.
data_test_pred <- data_test[,]  
data_test_pred$value_eur_pred <- Ypred # data_test + prediccion
data_test_pred$short_name <- df_fifa[-datos_indices, 'short_name']  # + short_name para hacerlo mas comprensible
data_test_pred[91:95, c('short_name', 'value_eur', 'value_eur_pred')]  # mostramos los 3 campos mas relevantes


# PASO 5: mejora del rendimiento del modelo
# Vamos a intentar mejorar al ajuste del modelo.
# Probamos incluir una relacion no lineal entre las caracteristicas. Consideramos overall^2
df_fifa2 <- df_fifa[,] 
df_fifa2$overall2 <- df_fifa$overall^2
data_train2 <- df_fifa2[datos_indices, -1]
data_test2 <- df_fifa2[-datos_indices, -1]
# Rehacemos el modelo
modelo2 <- lm(formula = value_eur ~  ., data = data_train2) 
summary(modelo2)
# Hay una mejora clara con el nuevo modelo2.
# R-squared = 0.923
# Un 92.3% de la variacion de la variable dependiente (value_err) es explicada por el modelo2
# El 50% de los errores estan dentro de los valores del primer y tercer cuartil, por lo que la  mayoria de las predicciones 
# estaban entre 0.92 Meur por encima del valor real y 0.97 Meur por debajo.

# Sobre el poder de prediccion de cada variable ((*** indican que son predictivas): overall, age, wage_eur, overall2, 
# international_reputation, potential y ciertos valores de team_position. 
# Ya con Pr(>|t|) > 0.05 preferred_foot y la mayoria de valores de team_position no se consideran predictivas. 

# Aplicamos la prediccion
Ypred2 = predict(modelo2, data_test2)
# Muestra de la prediccion (columna value_eur-pred). Mostramos 5 casos.
data_test_pred2 <- data_test2[,]  
data_test_pred2$value_eur_pred <- Ypred2 # data_test + prediccion
data_test_pred2$short_name <- df_fifa2[-datos_indices, 'short_name']  # + short_name para hacerlo mas comprensible
data_test_pred2[91:95, c('short_name', 'value_eur', 'value_eur_pred')]  # mostramos los 3 campos mas relevantes

# Comparamos los valores estimados con los del primer modelo. Mostramos 5 casos.
data_test_compared <- data_test_pred[, c('short_name', 'value_eur', 'value_eur_pred')]
data_test_compared$value_eur_pred2 <- data_test_pred2$value_eur_pred
data_test_compared[91:95,]
# Se ve una mejora clara en el ajuste con el modelo2. Modelo2 es mejor en 4 de las 5 predicciones. 
'C:\Users\34684\Dropbox\MSE\INSO4\AA-1\ProyectoFinal'
# Probamos un tercer modelo incluyendo sobre el modelo2 las variables age^2 o wage_eur^2, pero no se mejora el resultado (R^2 = 0.92).
# Probamos eliminar la variable preferred_foot y el resultado permanece igual (R^2 = 0.92). Se podria eliminar 
# para simplificar el modelo final.
# Probamos a eliminar los outliers del modelo <outlierTest(modelo2)> y no hay mejora.

# Por tanto modelo final = modelo2. Resumen:
# summary(modelo2)
# Coeficientes
modelo2$coef
# R-squared = 0.923. Un buen ajuste. Estamos ya preparados para estimar el precio de un jugador.
summary(modelo2)$r.squared



# PARTE 2: CLASIFICADOR KNN
# El objetivo en este caso es clasificar un jugador segun su rago de precio (value_eur) con el metodo KNN.
# Creamos para ello una nueva variable (categoria) nivel_precio.
# Poder clasificar un jugador por su rango de precio es muy relevante para la actividad de compraventa de jugadores, 
# no solo para clubes de futbol sino tambien para fondos de inversion interesados en adquirir los derechos 
# (o un porcentaje) de jugadores.

# PASO 1: Carga de Datos. Dataset 'players_20.csv'. 
# No hace falta carga de datos. Partiremos del dataframe df_fifa 
# y crearemos un nuevo dataframe df_fifa_knn con una nueva variable nivel_precio a partir de value_eur: 
# Bajo (0-5 Meur), Medio (5-20 Meur) y Alto (>20 Meur).

# PASO 2: Exploracion y Preparacion de los datos

nivel_precio <- function(valor){
  if (valor < 5) {
    return ('Bajo')  	
  } else if (valor < 20) {
    return ('Medio')  	
  } else {
    return ('Alto')  	
  }
}

v_selected = c('overall', 'age','potential', 'preferred_foot', 'international_reputation', 'team_position', 'wage_eur')
df_fifa_knn <- df_fifa[v_selected]
df_fifa_knn$nivel_precio <- sapply(df_fifa$value_eur, nivel_precio)
head(df_fifa_knn)
str(df_fifa_knn)

# convertimos factors a numeric para el scale
df_fifa_knn$preferred_foot <- as.numeric(df_fifa_knn$preferred_foot)
df_fifa_knn$team_position <- as.numeric(df_fifa_knn$team_position)
# target (nivel_precio) as ordered factor (Bajo < medio < Alto)
df_fifa_knn$nivel_precio <- ordered(as.factor(df_fifa_knn$nivel_precio), levels = c('Bajo', 'Medio', 'Alto'))
str(df_fifa_knn)

# scale de los datos
df_fifa_knn_norm <- scale(df_fifa_knn[,-8]) # sin la variable target (nivel_precio)
head(df_fifa_knn_norm)

# Cogemos 70% datos para train y 30% para test
# Uso la misma variable datos_indices que en la Parte 1
data_train_knn <- df_fifa_knn_norm[datos_indices,]
dim(data_train_knn)
data_test_knn <- df_fifa_knn_norm[-datos_indices,]
dim(data_test_knn)

data_train_knn_target <- df_fifa_knn[datos_indices, 8]
data_test_knn_target <- df_fifa_knn[-datos_indices, 8]
table(data_train_knn_target)
table(data_test_knn_target)
# Se ve que sample ha distribuido bien los datos entre train y test

# PASO 3: Entrenar el modelo con los datos
library(class)
my_k=5 # valor inicial de prueba
data_test_knn_pred <- knn(data_train_knn, data_test_knn, cl=data_train_knn_target, k=my_k)
table(data_test_knn_pred)

# PASO 4: Evaluar el rendimiento del modelo
library(caret)
print(paste("K =", my_k))
tab <- table(data_test_knn_pred, data_test_knn_target)
confusionMatrix(tab)
# K = 5, resultado bastante bueno: kappa 0.8594 y accuracy: 0.9248

# PASO 5: Mejora del rendimiento del modelo
# Probamos con diferentes valores de k: 3, 7, 10, 15, 20
print("confusionMatrix para K = 3, 7, 10, 15, 20")
for (my_k in c(3, 7, 10, 15, 20)) {
	print(paste("K =", my_k))
	data_test_knn_pred <- knn(data_train_knn, data_test_knn, cl=data_train_knn_target, k=my_k)
	tab <- table(data_test_knn_pred, data_test_knn_target)
	print(confusionMatrix(tab))
}


# K = 7 da el mejor resultado. kappa 0.8621 y accuracy: 0.9266
#data_test_knn_pred Bajo Medio Alto
#             Bajo   914    49    0
#             Medio   31   552   35
#             Alto     0     7   75

# El modelo resultante es muy satisfactorio y una buena herramienta de clasificacion. 





