# Modo Interactivo de Dibujo de Polígono

## 🎯 Vista General

¡Nuevo! Ahora puedes **dibujar polígonos manualmente punto por punto** en lugar de usar inicialización automática ConvexHull.

Workflow:
```
Selecciona [Polygon ROI] → Click para colocar vértices → ENTER para terminar
```

---

## 📖 Cómo Usar

### Paso 1: Selecciona Polygon ROI
```
En la sección ROI, haz click en: [Polygon]
```

### Paso 2: Haz Click en Scatter
```
Click en el botón "Scatter"
→ Se activa el modo de dibujo
→ Aparece una etiqueta amarilla con instrucciones
```

### Paso 3: Dibuja tu Polígono
```
Haz clicks en el gráfico para colocar vértices
- Cada click agrega un vértice
- Verás una línea conectando los puntos en tiempo real
- La etiqueta muestra: "Vertices: N | ENTER para terminar | ESC para cancelar"
```

### Paso 4: Termina el Dibujo
```
Presiona ENTER
- Polígono se cierra automáticamente
- Se convierte en un ROI editable
- Listo para clustering
```

### Paso 5: Opcional - Edita el Polígono
```
Si necesitas ajustar:
- Arrastra vértices para moverlos
- Right-click en borde para agregar vértices
- Ctrl+Click en vértice para removerloa
```

---

## 🔑 Controles Principales

| Acción | Comando |
|--------|---------|
| Agregar vértice | Click en el gráfico |
| Terminar dibujo | ENTER |
| Cancelar dibujo | ESC |
| Ver estado | Etiqueta amarilla |

---

## 💡 Consejos Prácticos

### Consejo 1: Planifica tu Trazo
```
Antes de empezar, identifica los puntos clave del axón
- Traza mental del contorno
- ~8-15 vértices es típicamente suficiente
```

### Consejo 2: Precisión vs Velocidad
```
Rápido (5-8 vértices):
- Cubre el axón con puntos clave
- Menos clicks = más rápido

Preciso (15-20 vértices):
- Más puntos en curvas
- Más trabajo de dibujo pero mejor forma
```

### Consejo 3: Usa Zoom si Necesitas Precisión
```
- Zoom in en la región de interés
- Click para ver la forma con más detalle
- Luego dibuja tu polígono con precisión
```

### Consejo 4: No Tengas Miedo de Equivocarte
```
Si cometes un error:
- Opción 1: Presiona ESC, comienza de nuevo
- Opción 2: Termina (ENTER), luego edita vértices
- Ninguna es "incorrecta", ambas funcionan
```

---

## Ejemplo Paso a Paso: Trazar un Axón Simple

### Escenario
Tienes un axón recto con dos bifurcaciones

### Solución
```
1. Carga datos → Click "Scatter"
2. Selecciona [Polygon] → Click "Scatter" nuevamente
3. Modo de dibujo activado ✨
4. Clicks para dibujar:
   - Punto 1: Base del axón
   - Punto 2: Lado izquierdo arriba
   - Punto 3: Primera bifurcación (derecha)
   - Punto 4: Segunda bifurcación (derecha)
   - Punto 5: Lado derecho abajo
   - Punto 6: Lado derecho arriba
   - Punto 7: Vuelve cerca del punto 1
5. Presiona ENTER → Polígono completo
6. Si necesita ajuste, edita vértices
7. Click "Cluster on Ch1"
```

---

## ⚠️ Limitaciones & Notas

| Limitación | Detalles |
|------------|----------|
| Mínimo 3 vértices | Los polígonos válidos requieren al menos 3 puntos |
| Máximo ~100 vértices | Por arriba se vuelve lento (pero técnicamente funciona) |
| No automático | Requiere user clicks (a diferencia del ConvexHull anterior) |
| Edición posterior | Los vértices siguen siendo editables después de terminar |

---

## 🔧 Solución de Problemas

### P: No veo nada cuando hago click
**A**: Asegúrate que:
1. ✅ Datos estén cargados (scatter plot muestra puntos)
2. ✅ [Polygon] esté seleccionado
3. ✅ Clicks dentro del área del gráfico

### P: Presioné ENTER pero nada pasó
**A**: Debes tener **mínimo 3 vértices**. Verifica que hayas hecho al menos 3 clicks.

### P: Quiero volver a dibujar
**A**: Presiona **ESC** para cancelar. Luego click "Scatter" de nuevo para empezar.

### P: ¿Puedo editar el polígono después?
**A**: Sí. Después de ENTER, puedes:
- Arrastrar vértices
- Right-click para agregar
- Ctrl+click para remover
Luego click "Cluster" para usar la versión editada.

### P: ¿Por qué mi polígono se ve raro?
**A**: Verifica que:
1. Orden de vertices (no se cruzen los lados)
2. Polígono cierre completamente alrededor del axón
3. Si es muy complicado, simplifica (menos vértices)

---

## 🎬 Workflow Completo

```
┌─ INICIO ─────────────────┐
│ Carga datos              │
│ Click "Scatter"          │
└────────────┬─────────────┘
             │
             ↓
┌─ DIBUJAR ────────────────┐
│ [Polygon] radio button   │
│ Click "Scatter" de nuevo │
│ → Modo de dibujo activo  │
└────────────┬─────────────┘
             │
             ↓
┌─ CLICKS ─────────────────┐
│ Click para cada vértice  │
│ Polyline se actualiza    │
│ Etiqueta muestra estado  │
└────────────┬─────────────┘
             │
             ↓
┌─ TERMINAR ────────────────┐
│ Presiona ENTER           │
│ Polígono se cierra       │
│ ROI listo para editar    │
└────────────┬──────────────┘
             │
             ↓
┌─ CLUSTER ────────────────┐
│ Click "Cluster on Ch1"   │
│ Ray-casting filtra datos │
│ Resultados muestran      │
└──────────────────────────┘
```

---

## ⏱️ Performance

| Acción | Tiempo |
|--------|--------|
| Dibujar 10 vértices | ~10 segundos |
| Filtrado con 50 vértices | ~200ms |
| Filtrado con 100 vértices | ~500ms |
| Clustering posterior | Normal (sin cambios) |

---

## 🎓 Comparación: Métodos de Creación de Polígono

### Método 1: Dibujo Interactivo (NUEVO - Recomendado)
```
✅ Control total sobre la forma
✅ Intuitive (click para cada punto)
✅ Ver polyline en tiempo real
✅ ENTER para terminar
❌ Requiere trabajo manual
❌ 15-30 segundos de dibujo
```

### Método 2: ConvexHull Automático (Anterior)
```
✅ Automático (rápido)
✅ Buena aproximación inicial
❌ Puede no coincidir exactamente
❌ Requiere edición posterior
```

---

## 📝 Notas de Implementación

- **Motor**: PyQtGraph's `PolyLineROI` + mouse events
- **Algoritmo**: Ray-casting vectorizado (NumPy)
- **Compatibilidad**: 100% con filtering y clustering existente
- **Fallback**: Si ESC presionado, vuelve a ROI circular

---

## 🚀 Próximas Características (Futuro)

- [ ] Guardar polígonos como templates
- [ ] Cargar polígonos guardados
- [ ] Polígonos múltiples (unión/intersección)
- [ ] Estadísticas del polígono (área, perímetro)
- [ ] Suavizado de curvas en tiempo real

---

**¡Ahora puedes trazar los axones exactamente como los ves!** 🎯
