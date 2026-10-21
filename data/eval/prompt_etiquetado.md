# Prompt para etiquetar el set de evaluación

Este prompt se le da a un modelo de IA **distinto** al que categoriza en la app, para que las
etiquetas no coincidan con los criterios de quien escribió el prompt de categorización.
La salida se guarda como `data/eval/transacciones_etiquetadas.csv` y luego se revisa a mano.

---

Necesito armar un set de evaluación de 100 movimientos de estados de cuenta bancarios de
personas en Lima, Perú, cada uno con su categoría correcta. Los datos son sintéticos: no uses
datos reales de personas.

## Categorías (usa exactamente estos nombres)

- Comida y delivery
- Supermercado
- Transporte
- Suscripciones
- Entretenimiento
- Salud
- Educación
- Servicios (luz, agua, internet, celular)
- Transferencias
- Ingresos
- Otros

No te doy definiciones a propósito: usa tu propio criterio, como lo haría una persona que
revisa su estado de cuenta. Los montos negativos son cargos (sale dinero) y los positivos son
abonos (entra dinero).

## Parte 1: etiqueta estos 50 movimientos

Copia `descripcion` y `monto` exactamente como están. Prefijos como IZI, MERPAGO, NIUBIZ,
CULQI, VENDEMAS, PAYPAL, DLC o POS son pasarelas o terminales de pago, no el comercio.

```csv
descripcion,monto
MERPAGO*LIBRERIA EL ESTUDIANTE,-52.94
TRANSFERENCIA RECIBIDA M QUISPE,450.00
MIFARMA MIRAFLORES 53581,-21.92
RAPPI*PERU SAC 5165,-16.34
VENDEMAS*JUGUERIA LA FRUTA,-14.09
NIUBIZ*BARBERIA EL CORTE,-29.56
PEDIDOSYA*PEDIDO 92506,-28.06
IZI*INKAFARMA 74000,-25.37
RAPPI*PERU SAC 72482,-18.91
TAMBO+ LARCO 80840,-17.33
TRANSFERENCIA RECIBIDA M QUISPE,450.00
TOTTUS JOCKEY PLAZA 28460,-88.67
PLIN-CARLA MENDOZA,-31.56
YAPE-LUIS PAREDES,-16.98
INDRIVE*VIAJE 83397,-15.04
IZI*POLLERIA LA BRASA,-34.26
PLIN-CARLA MENDOZA,-33.38
YAPE-ANA TORRES,-37.69
POS PLAZA VEA SURCO 75364,-143.63
TRANSFERENCIA RECIBIDA M QUISPE,450.00
IZI*POLLERIA LA BRASA,-34.86
PAYPAL *SPOTIFY,-21.90
BODEGA DON LUCHO,-8.25
MOVISTAR MOVIL PLAN POSTPAGO,-59.90
PEDIDOSYA*PEDIDO 90166,-39.39
RAPPI*PERU SAC 83240,-22.20
POS WONG BENAVIDES 48576,-163.02
IZI*COPIAS Y MAS,-10.72
MERPAGO*CINEPLANET 86180,-26.20
TAMBO+ LARCO 8685,-7.46
MERPAGO*CINEPLANET 39890,-21.53
RAPPI*RAPPI PRO,-24.90
VENDEMAS*JUGUERIA LA FRUTA,-8.25
DLC*NETFLIX.COM,-44.90
VENDEMAS*JUGUERIA LA FRUTA,-7.33
PEDIDOSYA*PEDIDO 62993,-42.68
RAPPI*PERU SAC 44309,-18.19
INDRIVE*VIAJE 19373,-12.48
CULQI*GIMNASIO FORMA FIT,-86.52
PEDIDOSYA*PEDIDO 13363,-15.68
PEDIDOSYA*PEDIDO 47438,-28.10
YAPE-ANA TORRES,-13.98
OPENAI *CHATGPT SUBSCR,-75.00
TAMBO+ LARCO 96748,-24.90
VENDEMAS*JUGUERIA LA FRUTA,-14.14
IZI*CHIFA DRAGON DORADO,-24.65
IZI*POLLERIA LA BRASA,-37.00
UBER *TRIP 13899,-9.33
CULQI*GIMNASIO FORMA FIT,-82.48
RAPPI*PERU SAC 71678,-20.95
```

## Parte 2: genera 50 movimientos nuevos y etiquétalos

Escríbelos con el estilo de un estado de cuenta peruano: mayúsculas, a veces con prefijo de
pasarela (IZI*, MERPAGO*, NIUBIZ*, CULQI*, PAYPAL *) y a veces con un número de operación al
final. Montos realistas en soles, con dos decimales.

- **20 movimientos** de estos comercios (uno o dos por comercio, variados): UBER ONE, DISNEY,
  PRIME VIDEO, HBO, YOUTUBE PREMIUM, ICLOUD, UBER EATS, DIDI FOOD, KFC, BEMBOS, PIZZA HUT,
  STARBUCKS, METRO, VIVANDA, MAKRO, OXXO, CABIFY, DIDI, METROPOLITANO, PRIMAX, CINEMARK,
  TELETICKET, JOINNUS, STEAM, BOTICAS, CLINICA, PUCP, COURSERA, PLATZI, CLARO, ENTEL, BITEL,
  SEDAPAL, LUZ DEL SUR, CALIDDA, TUNKI.
- **30 movimientos** de comercios o conceptos que **no contengan como palabra completa**
  ninguna de estas palabras: AMAZON, BEMBOS, BITEL, BOTICA, BOTICAS, CABIFY, CALIDDA, CHATGPT, CINEMARK, CINEPLANET, CINEPOLIS, CLARO, CLINICA, COURSERA, CRUNCHYROLL, DEL, DIDI, DISNEY, EATS, ENEL, ENTEL, FARMACIA, FOOD, GOOGLE, HABERES, HBO, HUT, ICLOUD, INDRIVE, INKAFARMA, JOHNS, JOINNUS, KFC, LUZ, MAKRO, MASS, MCDONALDS, METRO, METROPOLITANO, MIFARMA, MOVISTAR, NETFLIX, NORKYS, ONE, OPENAI, OXXO, PAPA, PEDIDOS, PEDIDOSYA, PENSION, PIZZA, PLANILLA, PLATZI, PLAYSTATION, PLAZA, PLAZAVEA, PLIN, PREMIUM, PRIMAX, PRIME, PRO, PUCP, RAPPI, REMUNERACION, REPSOL, SEDAPAL, SPOTIFY, STARBUCKS, STEAM, SUELDO, SUR, TAMBO, TELETICKET, TICKETMASTER, TOTTUS, TRANSF, TRANSFERENCIA, TUNKI, UBER, UDEMY, UNIVERSIDAD, UPC, UTEC, VEA, VIDEO, VIVANDA, WONG, YA, YAPE, YOUTUBE.
  Por ejemplo: cevicherías, pollerías, mercados, ferreterías, peluquerías, gimnasios,
  veterinarias, librerías, pagos de impuestos, academias, cines independientes, peajes,
  estacionamientos, apps poco conocidas. Incluye **al menos 5 abonos** (montos positivos que
  no sean sueldo, por ejemplo reembolsos o pagos de un amigo) y **al menos 8 casos que
  consideres ambiguos**.

## Formato de salida

Devuelve **solo** un bloque CSV con exactamente 100 filas más la cabecera:

```
descripcion,monto,categoria_esperada,segmento,ambigua,nota
```

- `segmento`: `catalogo` para las 50 de la Parte 1 y `fuera` para las 50 de la Parte 2.
- `ambigua`: `si` si una persona razonable podría elegir otra categoría; si no, `no`.
- `nota`: solo en las ambiguas, una frase corta con la otra categoría posible. Si no, vacío.
- Pon entre comillas cualquier campo que tenga comas.
