# Dentix – Análisis Multivariado y Modelos Predictivos

**Aplicación de Métodos Multivariados para la Ciencia de Datos**  
Proyecto académico orientado al análisis de riesgo crediticio para Clínicas Dentix. Este repositorio contiene el análisis estadístico, modelado predictivo, interpretabilidad y segmentación necesarios para construir un **score de mora** y un **modelo de monto aprobado**, utilizando datos reales de créditos otorgados por Dentix.

---

## 1. Contexto del Proyecto

Las Clínicas Dentix otorgan créditos para financiar tratamientos odontológicos. Se observa:

- Variación en la **mora** entre clientes, clínicas y asesores.
- Aprobaciones heterogéneas.
- Necesidad de estandarizar políticas de riesgo y priorización de cobranza.

El objetivo central es anticipar, desde el origen del crédito, **la probabilidad de caer en mora** y analizar **qué factores explican el monto aprobado**, dentro de un marco multivariado riguroso.

---

## 2. Objetivos del Proyecto

### **Modelos predictivos**

1. **Mora ≥ 30 días**

   - Score de riesgo mediante **Regresión Logística**.
   - Métricas: AUC, KS, matriz de confusión, sensibilidad/especificidad.

2. **Franjas de mora (ordinal)**

   - Modelo ordinal y evaluación mediante MAE ordenado.

3. **Monto aprobado**
   - Regresión lineal / robusta / regularizada.
   - Revisión de heteroscedasticidad y análisis de residuos.

### **Análisis explicativo**

- Variables que más influyen en el monto aprobado.
- Segmentos altamente morosos mediante MANOVA, análisis discriminante y clustering.
- Comparación de riesgo entre **clínicas** y **asesores**.

### **Recomendaciones**

- Política costo–sensitiva de aprobación.
- Priorización de cobranza.
- Escenarios con riesgo controlado.
