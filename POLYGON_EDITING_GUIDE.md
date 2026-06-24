# Cómo Marcar/Editar el Polígono ROI

## Resumen Rápido

Cuando seleccionas **"Polygon ROI"** y haces click en "Scatter", un polígono inteligente aparece automáticamente. Aquí está cómo editarlo:

---

## Controles Interactivos

### 1. **Arrastrar Vértices** (Lo más importante)
```
1. Pasa el cursor sobre cualquier vértice (punto) del polígono
2. Haz click y arrastra ese vértice a la nueva posición
3. El polígono se actualiza en tiempo real
```
✨ **Esto es lo que usarás más para ajustar la forma**

### 2. **Mover Polígono Completo**
```
1. Haz click en el centro del polígono
2. Arrastra para mover todo el polígono
3. Todos los vértices se mueven juntos
```

### 3. **Agregar Vértices Nuevos**
```
1. Mantén presionada la tecla CTRL
2. Haz click en cualquier punto del borde del polígono
3. Se crea un nuevo vértice en ese punto
4. Arrastra inmediatamente para posicionarlo
```

**Cuándo usarlo**: Cuando necesites más detalle en una área (ej. una curva pronunciada)

### 4. **Remover Vértices**
```
1. Mantén presionada la tecla SHIFT
2. Haz click en el vértice que quieres remover
3. El vértice desaparece
```

**Requisito**: Debes mantener al menos 3 vértices (triángulo mínimo)

---

## Ejemplo Práctico: Trazar un Axón

### Paso a Paso

**1. Polygon aparece** (ConvexHull automático)
```
Ves un polígono con ~10-20 vértices
Approximadamente alrededor de tu axón
```

**2. Ajusta los vértices principales**
```
- Identifica las partes que no coinciden con el axón
- Arrastra esos vértices para que encajen mejor
- Usa 3-5 ajustes principales
```

**3. Agrega vértices en curvas**
```
- Si tu axón tiene una curva pronunciada
- CTRL+Click en el borde para agregar un vértice
- Arrastra ese vértice hacia la curva
```

**4. Refina detalles**
```
- Ajusta cualquier vértice que no encaje perfectamente
- Agrega más si necesitas más precisión
- Remueve vértices que no necesites
```

**5. Verifica**
```
- El polígono debe rodear completamente el axón
- Ningún espacio grande entre polígono y axón
- Listo para Cluster!
```

---

## Consejos Prácticos

### Consejo 1: Empieza Grueso, Refina Después
```
No intentes ser perfecto en el primer intento
- Ajusta los 5-10 vértices principales primero
- Luego agrega vértices en áreas que necesiten más detalle
- Más rápido y menos confuso
```

### Consejo 2: Usa CTRL+Click Estratégicamente
```
No hagas un vértice en cada punto
- Agrega vértices solo donde la curva cambia mucho
- En secciones rectas, 2-3 vértices son suficientes
- En curvas, coloca vértices donde el ángulo cambia
```

### Consejo 3: Zoom In
```
Si la precisión es crítica:
- Usa el zoom de PyQtGraph para acercar
- Esto te da mejor control visual
- Especialmente útil para detalles finos
```

### Consejo 4: Prueba y Ajusta
```
Después de hacer Cluster:
1. Si el resultado no es bueno, edita el polígono más
2. Haz Cluster de nuevo
3. Los cambios son instantáneos - no hay penalidad
```

---

## Comando Summary (Teclado + Ratón)

| Acción | Comando |
|--------|---------|
| Mover vértice | Click + Arrastrar en vértice |
| Mover polígono | Click + Arrastrar en centro |
| Agregar vértice | CTRL + Click en borde |
| Remover vértice | SHIFT + Click en vértice |

---

## Solución de Problemas

### Problema: No puedo seleccionar un vértice
**Solución**: 
- Asegúrate de hacer click exactamente en el vértice
- Si el polígono está muy pequeño, usa zoom in
- Si aún no funciona, intenta hacer click más cerca del centro del vértice

### Problema: CTRL+Click no agrega vértice
**Solución**:
- Verifica que CTRL esté siendo presionado (puede variar por sistema operativo)
- En algunos sistemas, podría ser ALT o CMD
- Intenta hacer click directamente en el borde del polígono (no en el espacio vacío)

### Problema: El polígono se comporta extraño
**Solución**:
- Recarga el scatter plot (vuelve a clickear "Scatter")
- Selecciona "Polygon ROI" de nuevo
- El polígono debería crearse fresco

### Problema: He hecho demasiados cambios y se ve mal
**Solución**:
- Opción 1: Recrea el ROI (click en "Scatter" nuevamente)
- Opción 2: Remueve vértices innecesarios (SHIFT+Click)
- Opción 3: Inicia una nueva sesión con los datos

---

## Cuándo Haces Cluster

Una vez que el polígono se vea bien:

```
1. Verifica que rodee completamente el axón
2. Establece Z-min y Z-max si necesitas filtrado de profundidad
3. Haz click "Cluster on Ch1" o "Cluster on Ch2"
4. El sistema usa ray-casting para filtrar puntos dentro del polígono
5. Clustering procede normalmente con los puntos filtrados
```

**Tiempo esperado para filtrado**: < 1 segundo incluso con 100 vértices y 1M puntos

---

## Preguntas Frecuentes

**P: ¿Puedo tener un polígono con 200 vértices?**
A: Técnicamente sí, pero se vuelve lento. 50-100 vértices es el rango práctico.

**P: ¿Puedo guardar el polígono para usarlo de nuevo?**
A: Actualmente no, pero es una feature futura planeada.

**P: ¿Qué pasa si cometo un error mientras edito?**
A: Simplemente edita más. Los cambios se aplican inmediatamente a la próxima vez que hagas Cluster.

**P: ¿Es mejor muchos vértices o pocos?**
A: Un equilibrio. Suficientes para capturar la forma, pero no tantos que sea imposible editar.

---

## Próximos Pasos

Una vez que domines la edición del polígono:

1. ✅ Practica con datos conocidos
2. ✅ Prueba con polígonos complejos (L-shaped, etc)
3. ✅ Experimenta con spline smoothing (curvas suaves)
4. ✅ Usa múltiples ROI diferentes en el mismo dataset

---

**¡Buena suerte marcando tus axones!** 🎯
