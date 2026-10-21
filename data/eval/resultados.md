# Resultados de la evaluación (21/10/2026, modelo claude-haiku-5-5)

## Categorización

- Transacciones: 100 (50 del catálogo, 50 fuera del catálogo; 20 marcadas como ambiguas)
- Corridas: 3

| Métrica | Accuracy |
|---|---|
| Global | 90.3% (rango 89.0%–91.0%) |
| Global sin ambiguas | 94.6% (rango 93.8%–95.0%) |
| Segmento «catalogo» (n=50) | 94.0% (rango 94.0%–94.0%) |
| Segmento «fuera» (n=50) | 86.7% (rango 84.0%–88.0%) |
| Capa «reglas» (n=56, 1.ª corrida) | 91.1% (rango 91.1%–91.1%) |
| Capa «llm» (n=38, 1.ª corrida) | 87.7% (rango 84.2%–89.5%) |
| Capa «cache» (n=6, 1.ª corrida) | 100.0% (rango 100.0%–100.0%) |

**Por categoría (1.ª corrida)**

| Categoría | n | Accuracy |
|---|---|---|
| Comida y delivery | 28 | 85.7% |
| Transferencias | 12 | 91.7% |
| Suscripciones | 11 | 81.8% |
| Otros | 9 | 100.0% |
| Entretenimiento | 8 | 87.5% |
| Transporte | 8 | 100.0% |
| Supermercado | 7 | 100.0% |
| Servicios (luz, agua, internet, celular) | 6 | 83.3% |
| Salud | 6 | 100.0% |
| Educación | 4 | 100.0% |
| Ingresos | 1 | 100.0% |

**Errores (1.ª corrida)**

| Descripción | Esperada | Predicha | Capa | Ambigua |
|---|---|---|---|---|
| TAMBO+ LARCO 80840 | Comida y delivery | Supermercado | reglas | sí |
| TAMBO+ LARCO 8685 | Comida y delivery | Supermercado | reglas | sí |
| TAMBO+ LARCO 96748 | Comida y delivery | Supermercado | reglas | sí |
| OXXO BENAVIDES 39102 | Comida y delivery | Supermercado | reglas | sí |
| CULQI*PLATZI SUBSCR | Suscripciones | Educación | reglas | no |
| CULQI*SMARTFIT MEMBRESIA | Suscripciones | Salud | llm | no |
| CRISOL OVALO GUTIERREZ 10293 | Entretenimiento | Otros | llm | sí |
| JUNTA PROPIETARIOS EDIFICIO | Servicios (luz, agua, internet, celular) | Otros | llm | no |
| BIM-ROBERTO GOMEZ | Transferencias | Otros | llm | no |

## Chat

- Preguntas: 15 (incluye 3 que no se pueden responder) · corridas: 3
- Respuestas correctas: 100.0% (rango 100.0%–100.0%)

| # | Pregunta | Correctas | Herramientas (1.ª corrida) |
|---|---|---|---|
| 1 | ¿Cuánto gasté en total en septiembre? | 3/3 | resumen_mensual |
| 2 | ¿Cuánto ingresé en agosto? | 3/3 | resumen_mensual |
| 3 | ¿Cuál fue mi balance en julio? | 3/3 | resumen_mensual |
| 4 | ¿Cuánto varió mi gasto de agosto a septiembre? | 3/3 | resumen_mensual, resumen_mensual |
| 5 | ¿Cuál fue mi gasto más grande y cuándo fue? | 3/3 | top_gastos |
| 6 | ¿Cuánto gasté en Uber en total? | 3/3 | buscar_transacciones |
| 7 | ¿Cuántas veces pedí por PedidosYa? | 3/3 | buscar_transacciones |
| 8 | ¿Cuál fue mi pedido de Rappi más caro? | 3/3 | buscar_transacciones |
| 9 | ¿Cuándo fue mi último viaje en Uber? | 3/3 | listar_transacciones |
| 10 | ¿Cuánto gasté en Wong en total? | 3/3 | buscar_transacciones |
| 11 | ¿Cuánto me cobra Netflix al mes? | 3/3 | buscar_transacciones, listar_suscripciones |
| 12 | ¿Cuánto gasté en total en julio? | 3/3 | resumen_mensual |
| 13 | ¿Cuánto gasté en marzo de 2026? | 3/3 | - |
| 14 | ¿A cuánto está el tipo de cambio del dólar hoy? | 3/3 | - |
| 15 | ¿En qué acciones debería invertir mis ahorros? | 3/3 | - |

_Costo total de la evaluación: US$ 0.0200 (87 llamadas, 154,791 tokens de entrada, 9,102 de salida)._
