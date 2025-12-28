# Configuración para Kaggle

Este documento describe cómo ejecutar el proyecto CIS Factual LLM en Kaggle.

## Pasos para configurar en Kaggle

### 1. Crear un nuevo notebook en Kaggle

Ve a [Kaggle Notebooks](https://www.kaggle.com/code) y crea un nuevo notebook.

### 2. Configurar el acelerador GPU

- En la parte derecha, bajo "Settings", activa **GPU T4 x2** o **GPU P100**
- Esto es necesario para cargar el modelo Mistral-7B en FP16

### 3. Clonar el repositorio

En la primera celda del notebook, ejecuta:

```python
!git clone https://github.com/notGiGi/CIS.git
%cd CIS
```

### 4. Instalar dependencias

```python
!pip install -q -r requirements.txt
```

### 5. Flash Attention 2 (Opcional - No requerido)

Flash Attention 2 puede acelerar la inferencia, pero **no es necesario**. El código automáticamente detecta si no está disponible y usa atención estándar.

Si quieres intentar instalarlo (puede tardar varios minutos):

```python
!pip install -q flash-attn --no-build-isolation
```

**Nota:** La instalación puede fallar en Kaggle. Esto está bien - el modelo cargará automáticamente con atención estándar.

### 6. Descargar el modelo Mistral-7B

Kaggle tiene cache de Hugging Face, así que modifica el config:

En una celda Python:

```python
import yaml

# Leer configuración actual
with open('config/model.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Modificar para Kaggle
config['local_files_only'] = False  # Permitir descarga desde HF
config['cache_dir'] = '/kaggle/working/model_cache'  # Cache local

# Guardar
with open('config/model.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print("Configuración actualizada para Kaggle")
```

### 7. Ejecutar el experimento

```python
!python src/experiments/run_single_fact.py --config config/experiment.yaml
```

## Configuración de memoria

El modelo está configurado para usar toda la memoria GPU disponible. En Kaggle:

- **T4 (16GB)**: Funcionará perfectamente en FP16
- **P100 (16GB)**: Funcionará perfectamente en FP16
- **GPU T4 x2**: Usará distribución automática entre GPUs

## Ejemplo de notebook completo

```python
# Celda 1: Setup
!git clone https://github.com/notGiGi/CIS.git
%cd CIS
!pip install -q -r requirements.txt

# Celda 2: Configurar para Kaggle
import yaml

with open('config/model.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['local_files_only'] = False
config['cache_dir'] = '/kaggle/working/model_cache'

with open('config/model.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print("✓ Configuración lista para Kaggle")

# Celda 3: Ejecutar experimento
!python src/experiments/run_single_fact.py --config config/experiment.yaml

# Celda 4 (Opcional): Experimentos personalizados
from src.models.load_model import load_model_and_tokenizer
from src.utils.token_utils import get_next_token_logits, decode_topk_predictions
from src.prompts.factual_prompts import make_factual_prompt

# Cargar modelo
with open('config/model.yaml', 'r') as f:
    model_config = yaml.safe_load(f)

model, tokenizer = load_model_and_tokenizer(model_config)

# Tu experimento personalizado aquí
prompt = make_factual_prompt("Eiffel Tower", "located in")
logits = get_next_token_logits(model, tokenizer, prompt, device="cuda")
predictions = decode_topk_predictions(tokenizer, logits, k=10)

for i, pred in enumerate(predictions, 1):
    print(f"{i}. {pred['token_str']!r} - prob={pred['prob']:.4f}")
```

## Optimizaciones para Kaggle

### Si tienes problemas de memoria:

Modifica `config/model.yaml`:

```yaml
use_4bit: true  # Activar cuantización 4-bit
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: float16
bnb_4bit_use_double_quant: true
```

### Para acelerar la inferencia:

```yaml
use_flash_attention: true  # Ya está activado por defecto
```

## Archivos importantes

- `config/model.yaml`: Configuración del modelo
- `config/experiment.yaml`: Configuración del experimento
- `src/experiments/run_single_fact.py`: Script principal
- `data/counterfact_subset.json`: Datos de ejemplo

## Troubleshooting

### Error: "CUDA out of memory"

Solución: Activa cuantización 4-bit (ver arriba)

### Error: "Cannot download model"

Solución: Asegúrate de que `local_files_only: false` en el config

### Warning: "Flash Attention not available"

Esto es normal - el código usará atención estándar automáticamente.

## Recursos

- Repositorio: https://github.com/notGiGi/CIS
- Modelo: [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
