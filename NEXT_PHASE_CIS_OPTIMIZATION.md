# Fase 3: Optimización CIS (Counterfactual Internal States)

## 📋 Tabla de Contenidos
- [Qué es CIS Optimization](#qué-es-cis-optimization)
- [Diferencia con Causal Check](#diferencia-con-causal-check)
- [Objetivo de Esta Fase](#objetivo-de-esta-fase)
- [Cómo Funciona](#cómo-funciona)
- [Implementación](#implementación)
- [Métricas y Evaluación](#métricas-y-evaluación)
- [Archivos a Crear](#archivos-a-crear)
- [Flujo de Ejecución](#flujo-de-ejecución)

---

## 🎯 Qué es CIS Optimization

**CIS (Counterfactual Internal State)** es una perturbación mínima de activaciones internas que logra cambiar la predicción del modelo de un hecho verdadero a uno contrafactual.

### Ejemplo Concreto

**Hecho Real:**
- Prompt: `"The Eiffel Tower is located in"`
- Predicción: `" Paris"` (correcto)

**Objetivo CIS:**
- Encontrar el vector δ más pequeño posible que:
  - Agregado a la capa 16, posición -1
  - Cambie la predicción a `" Rome"` (contrafactual)
  - Tenga la norma L2 mínima

**Resultado:**
- δ optimizado con norma ||δ|| = 0.15 (por ejemplo)
- Esto mide qué tan "rígida" es la representación factual
- Norma pequeña = fácil de cambiar (débil rigidez factual)
- Norma grande = difícil de cambiar (fuerte rigidez factual)

---

## 🔄 Diferencia con Causal Check

| Aspecto | Causal Check (Fase 2) | CIS Optimization (Fase 3) |
|---------|----------------------|--------------------------|
| **Delta** | Aleatorio (Gaussian) | **Optimizado con gradientes** |
| **Objetivo** | Verificar que hooks funcionan | **Encontrar perturbación mínima** |
| **Target** | Ninguno (solo observar cambios) | **Target específico** (ej: " Rome") |
| **Optimización** | No hay | **Sí - gradient descent** |
| **Medición** | Cambio cualitativo | **Costo geométrico cuantitativo** |
| **Uso** | Sanity check | **Experimento científico real** |

### En Resumen

- **Fase 2 (Causal Check)**: "¿Funcionan los hooks?"
  - Delta = random
  - Solo verificamos que algo cambia

- **Fase 3 (CIS Optimization)**: "¿Cuál es el costo de cambiar este hecho?"
  - Delta = optimizado
  - Medimos rigidez factual

---

## 🎯 Objetivo de Esta Fase

### Pregunta Científica

**"¿Qué tan rígidas son las representaciones factuales en LLMs?"**

Específicamente:
1. ¿Cuál es la **perturbación mínima** necesaria para cambiar un hecho?
2. ¿Varía este costo entre diferentes hechos?
3. ¿Qué capas son más críticas para el conocimiento factual?

### Resultados Esperados

1. **Geometric Cost**: Norma L2 de la perturbación mínima
   - Bajo (< 0.1): Hecho débilmente codificado
   - Medio (0.1-1.0): Codificación normal
   - Alto (> 1.0): Hecho fuertemente arraigado

2. **Success Rate**: % de veces que logramos el flip
   - 100%: Siempre podemos cambiar la predicción
   - 50-99%: A veces funciona
   - < 50%: Hecho muy rígido o target inalcanzable

3. **Layer Sensitivity**: Qué capas son más efectivas
   - Hipótesis: Capas medias (12-20) más efectivas

---

## ⚙️ Cómo Funciona

### Pipeline Completo

```
1. Cargar modelo (Mistral-7B)
   ↓
2. Seleccionar hecho factual
   Ejemplo: "Eiffel Tower" → " Paris"
   ↓
3. Definir target contrafactual
   Ejemplo: " Rome"
   ↓
4. Inicializar delta (pequeño random o zeros)
   δ₀ ∈ ℝ^4096
   ↓
5. LOOP de optimización (N pasos):
   │
   ├─ 5a. Forward pass CON hook
   │      logits = model(prompt, con δ en capa L)
   │
   ├─ 5b. Calcular loss
   │      L = -log P(target) + λ·||δ||²
   │
   │      Donde:
   │      - P(target) = probabilidad del token target
   │      - λ = peso de regularización (ej: 0.01)
   │      - Queremos MAXIMIZAR P(target)
   │      - Queremos MINIMIZAR ||δ||
   │
   ├─ 5c. Backward pass
   │      ∇δ = ∂L/∂δ
   │
   ├─ 5d. Gradient descent
   │      δ ← δ - α·∇δ
   │      (α = learning rate, ej: 0.05)
   │
   └─ 5e. Check convergencia
        Si P(target) > umbral (ej: 0.5): STOP
   ↓
6. Evaluar resultado final
   - Costo geométrico: ||δ_final||
   - Success: ¿Es target el top-1?
   - Collateral: ¿Cambiaron otros tokens?
```

### Función de Loss

```python
def cis_loss(logits, target_token_id, delta, reg_weight=0.01):
    """
    Loss para CIS optimization.

    Args:
        logits: [vocab_size] logits del modelo
        target_token_id: ID del token que queremos
        delta: [hidden_size] perturbación actual
        reg_weight: peso de regularización L2

    Returns:
        loss: escalar para minimizar
    """
    # Probabilidad del target (queremos maximizarla)
    probs = torch.softmax(logits, dim=-1)
    target_prob = probs[target_token_id]

    # Loss principal: negative log likelihood
    # (minimizar = maximizar probabilidad)
    nll_loss = -torch.log(target_prob + 1e-10)

    # Regularización: penalizar norma grande de delta
    # (queremos delta pequeño)
    reg_loss = reg_weight * (delta ** 2).sum()

    # Loss total
    total_loss = nll_loss + reg_loss

    return total_loss
```

### ¿Por Qué Funciona?

1. **Backprop a través del modelo**: PyTorch calcula ∇δ automáticamente
2. **Hook modifica activaciones**: δ se suma en forward pass
3. **Gradiente nos dice**: "En qué dirección cambiar δ para aumentar P(target)"
4. **Regularización L2**: Evita que δ crezca descontroladamente

---

## 💻 Implementación

### Archivos a Crear

#### 1. `src/cis/cis_optimizer.py`

**Clase principal: `CISOptimizer`**

```python
class CISOptimizer:
    """Optimiza perturbaciones para lograr cambios contrafactuales."""

    def __init__(
        self,
        model,
        tokenizer,
        layer_idx: int,
        token_position: int = -1,
        device: str = "cuda"
    ):
        """
        Args:
            model: Transformer model (frozen)
            tokenizer: Tokenizer
            layer_idx: Qué capa intervenir
            token_position: Qué token perturbar (-1 = last)
            device: cuda/cpu
        """
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.token_position = token_position
        self.device = device
        self.hidden_size = get_hidden_size(model)

    def optimize(
        self,
        prompt: str,
        target_completion: str,
        max_steps: int = 200,
        learning_rate: float = 0.05,
        reg_weight: float = 0.01,
        tolerance: float = 1e-4,
        early_stop_margin: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Encuentra la perturbación mínima para lograr target.

        Returns:
            {
                'delta': torch.Tensor,          # Perturbación optimizada
                'final_loss': float,            # Loss final
                'geometric_cost': float,        # ||delta||
                'success': bool,                # ¿Logró el flip?
                'target_prob': float,           # P(target) final
                'num_steps': int,               # Pasos usados
                'top_predictions': List[Dict],  # Top-5 final
            }
        """
        # Implementación aquí...
```

**Métodos clave:**

```python
def _forward_with_intervention(self, input_ids, delta):
    """Forward pass con delta inyectado."""
    # 1. Attach hook con delta
    # 2. Run model
    # 3. Remove hook
    # 4. Return logits

def _compute_loss(self, logits, target_id, delta, reg_weight):
    """Calcula loss CIS."""
    # NLL + L2 regularization

def _backward_step(self, loss, delta, learning_rate):
    """Gradient descent step."""
    # 1. loss.backward()
    # 2. delta -= lr * delta.grad
    # 3. delta.grad.zero_()
```

#### 2. `src/experiments/run_cis_optimization.py`

**Script para experimentos CIS:**

```python
def run_cis_experiment(config_path: str):
    """
    Ejecuta optimización CIS en un hecho.

    1. Carga modelo
    2. Carga hecho de config o dataset
    3. Define target contrafactual
    4. Optimiza delta
    5. Reporta resultados
    """
    # Load model
    model, tokenizer = load_model_and_tokenizer(...)

    # Setup optimizer
    optimizer = CISOptimizer(
        model=model,
        tokenizer=tokenizer,
        layer_idx=config['layer'],
        token_position=config['token_position']
    )

    # Optimize
    result = optimizer.optimize(
        prompt="The Eiffel Tower is located in",
        target_completion=" Rome",
        max_steps=200,
        learning_rate=0.05,
    )

    # Report
    print(f"Success: {result['success']}")
    print(f"Geometric Cost: {result['geometric_cost']:.4f}")
    print(f"Target Probability: {result['target_prob']:.4f}")
```

#### 3. `src/metrics/factual_rigidity.py`

**Métricas para evaluar resultados:**

```python
def compute_geometric_cost(delta: torch.Tensor) -> float:
    """L2 norm of perturbation."""
    return delta.norm(p=2).item()

def compute_success_rate(results: List[Dict]) -> float:
    """Percentage of successful flips."""
    successes = sum(r['success'] for r in results)
    return successes / len(results)

def compute_collateral_effects(
    baseline_preds: List[str],
    intervention_preds: List[str]
) -> Dict[str, Any]:
    """Measure unintended changes."""
    # ¿Cuántos tokens en top-5 cambiaron?
    # ¿Qué tan diferente es la distribución?
```

#### 4. Actualizar `config/experiment.yaml`

```yaml
seed: 0
model_config: config/model.yaml
data_path: data/counterfact_subset.json

# Fact to test
subject: "Eiffel Tower"
relation: "located in"
expected_completion: " Paris"

# CIS Optimization
cis_optimization:
  target_completion: " Rome"     # Counterfactual target
  layer: 16                      # Which layer to intervene
  token_position: -1             # Which token (-1 = last)
  max_steps: 200                 # Max optimization steps
  learning_rate: 0.05            # Step size
  reg_weight: 0.01               # L2 regularization weight
  tolerance: 1.0e-4              # Convergence threshold
  early_stop_margin: 0.5         # Stop if P(target) > this

# Analysis
analysis:
  k_alternatives: 5              # Top-k to report
  measure_collateral: true       # Track side effects
```

---

## 📊 Métricas y Evaluación

### Métricas Principales

#### 1. **Geometric Cost** (Principal)
```python
cost = ||δ_optimized||_2
```
- **Interpretación**: Qué tan difícil es cambiar este hecho
- **Rango típico**: 0.01 - 10.0
- **Bajo (< 0.5)**: Fácil de cambiar, débil rigidez
- **Alto (> 2.0)**: Difícil de cambiar, fuerte rigidez

#### 2. **Success Rate**
```python
success = (top_1_prediction == target_token)
```
- **Interpretación**: ¿Logramos el flip?
- **100%**: Siempre exitoso
- **0%**: Nunca exitoso (target imposible o layer incorrecta)

#### 3. **Target Probability**
```python
target_prob = P(target | prompt, δ)
```
- **Interpretación**: Confianza del modelo en el target
- **> 0.5**: Target es top-1
- **< 0.1**: Target casi imposible

#### 4. **Optimization Convergence**
```python
num_steps_to_converge = steps_until(P(target) > threshold)
```
- **Interpretación**: Qué tan rápido converge
- **Pocos pasos (< 50)**: Fácil de optimizar
- **Muchos pasos (> 150)**: Difícil de optimizar

### Métricas Secundarias

#### 5. **Collateral Effects**
```python
collateral = |{tokens changed in top-5}| / 5
```
- **Interpretación**: Efectos secundarios de la intervención
- **0.0**: Solo cambió el target
- **1.0**: Toda la distribución cambió

#### 6. **Relative Rank Change**
```python
rank_change = rank_baseline(target) - rank_intervention(target)
```
- **Interpretación**: Cuánto subió el target en ranking
- **Ejemplo**: rank 50 → rank 1 = cambio de 49

---

## 📁 Archivos a Crear

### Estructura Completa

```
cis_factual_llm/
│
├── src/
│   ├── cis/
│   │   ├── __init__.py
│   │   └── cis_optimizer.py          # ← NUEVO: Clase CISOptimizer
│   │
│   ├── experiments/
│   │   ├── run_single_fact.py        # ✓ Ya existe
│   │   ├── run_causal_check.py       # ✓ Ya existe
│   │   └── run_cis_optimization.py   # ← NUEVO: Experimento CIS
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── factual_rigidity.py       # ← ACTUALIZAR: Métricas CIS
│   │
│   └── hooks/
│       └── residual_hooks.py         # ✓ Ya existe
│
├── config/
│   ├── model.yaml                    # ✓ Ya existe
│   └── experiment.yaml               # ← ACTUALIZAR: Config CIS
│
├── docs/
│   ├── CAUSAL_CHECK_GUIDE.md         # ✓ Ya existe
│   └── CIS_OPTIMIZATION_GUIDE.md     # ← NUEVO: Guía CIS
│
└── tests/
    ├── test_residual_hooks.py        # ✓ Ya existe
    └── test_cis_optimizer.py         # ← NUEVO: Tests CIS
```

---

## 🔄 Flujo de Ejecución

### Paso a Paso

#### 1. **Preparación**
```bash
# Verificar que causal check funciona
python src/experiments/run_causal_check.py --config config/experiment.yaml

# Output esperado:
# ✓ Causal effect detected
# ✓ Hook mechanism verified
```

#### 2. **Configurar Experimento CIS**
```yaml
# config/experiment.yaml
cis_optimization:
  target_completion: " Rome"    # Tu target contrafactual
  layer: 16                     # Capa media
  max_steps: 200
  learning_rate: 0.05
```

#### 3. **Ejecutar Optimización**
```bash
python src/experiments/run_cis_optimization.py --config config/experiment.yaml
```

#### 4. **Output Esperado**
```
================================================================================
CIS OPTIMIZATION: Counterfactual Internal State
================================================================================

Fact: "The Eiffel Tower is located in"
Baseline prediction: " Paris" (prob=0.823)
Target: " Rome"

Optimizing delta at layer 16, token position -1...

Step   0: Loss=5.234, P(target)=0.001, ||δ||=0.000
Step  10: Loss=3.456, P(target)=0.023, ||δ||=0.045
Step  20: Loss=2.123, P(target)=0.089, ||δ||=0.098
Step  30: Loss=1.234, P(target)=0.234, ||δ||=0.142
Step  40: Loss=0.789, P(target)=0.456, ||δ||=0.178
Step  50: Loss=0.456, P(target)=0.623, ||δ||=0.189  ← Target is top-1!

✓ Optimization converged in 50 steps

================================================================================
RESULTS
================================================================================

Success: ✓ YES
Geometric Cost: 0.189
Target Probability: 0.623
Convergence: 50 steps

Top-5 predictions (with intervention):
  1. " Rome"      prob=0.623  ← TARGET ✓
  2. " Paris"     prob=0.201
  3. " France"    prob=0.089
  4. " Italy"     prob=0.045
  5. " Europe"    prob=0.023

Collateral Effects: 4/5 tokens changed
Relative Rank Change: 49 (rank 50 → rank 1)

================================================================================
INTERPRETATION
================================================================================

✓ Successfully flipped "Paris" → "Rome"
✓ Geometric cost = 0.189 (moderate rigidity)
✓ Converged quickly (50 steps)

This fact has MODERATE factual rigidity:
- Not too easy to change (cost > 0.1)
- Not too hard to change (cost < 1.0)
- Middle layers effective for intervention
```

---

## 🎓 Conceptos Clave

### ¿Qué es "Geometric Cost"?

**Intuición geométrica:**

Imagina el espacio de activaciones como un paisaje de N dimensiones (N=4096 para Mistral).

- **Punto A**: Activaciones que predicen " Paris" (baseline)
- **Punto B**: Activaciones que predicen " Rome" (target)
- **Distancia A→B**: Geometric cost

```
Espacio de activaciones (4096D):

        " Paris"              " Rome"
           🗼                    🏛️
           │ ◄──── δ ────►      │
           A                    B

||δ|| = distancia más corta de A a B
```

### ¿Qué aprenderemos?

1. **Rigidez Factual Global**
   - Promedio de costs sobre muchos hechos
   - "¿Qué tan estables son los hechos en general?"

2. **Varianza entre Hechos**
   - Algunos hechos: cost bajo (fácil cambiar)
   - Otros hechos: cost alto (difícil cambiar)
   - "¿Hay hechos más 'arraigados' que otros?"

3. **Sensibilidad de Capas**
   - Comparar costs en diferentes layers
   - "¿Qué capas codifican conocimiento factual?"

4. **Targets Alcanzables**
   - Algunos targets: fácil alcanzar
   - Otros targets: imposible alcanzar
   - "¿Qué limita los cambios contrafactuales?"

---

## 🚀 Siguientes Pasos

### Implementación Incremental

#### Fase 3.1: CIS Optimizer Básico
```python
# Solo lo esencial
- Clase CISOptimizer
- Método optimize() básico
- Loss = NLL + L2 regularization
- Sin early stopping avanzado
```

#### Fase 3.2: Experimento Single-Fact
```python
# Probar en un hecho
- run_cis_optimization.py
- Config YAML
- Output detallado
```

#### Fase 3.3: Métricas y Análisis
```python
# Evaluar resultados
- Geometric cost
- Success rate
- Collateral effects
```

#### Fase 3.4: Batch Processing
```python
# Escalar a múltiples hechos
- Loop sobre dataset
- Guardar resultados en JSON
- Estadísticas agregadas
```

---

## 📚 Referencias

### Papers Relevantes

1. **Causal Tracing** (Meng et al., 2022)
   - Locating factual knowledge in LLMs
   - Layer-wise attribution

2. **ROME** (Meng et al., 2022)
   - Rank-One Model Editing
   - Similar idea: minimal edits to change facts

3. **Activation Engineering** (Turner et al., 2023)
   - Steering model behavior with activations
   - Diferentes objetivos pero técnica similar

### Recursos Técnicos

- **PyTorch Autograd**: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
- **Hook API**: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
- **Optimizer Implementation**: https://pytorch.org/docs/stable/optim.html

---

## ⚠️ Consideraciones Importantes

### Hiperparámetros Críticos

1. **Learning Rate** (`0.05` default)
   - Muy bajo (< 0.01): Convergencia lenta
   - Muy alto (> 0.5): Inestabilidad, overshoot
   - **Recomendación**: Empezar con 0.05, ajustar si no converge

2. **Regularization Weight** (`0.01` default)
   - Muy bajo (< 0.001): Delta crece mucho, cost alto
   - Muy alto (> 0.1): No alcanza el target
   - **Recomendación**: 0.01 para balance costo/éxito

3. **Max Steps** (`200` default)
   - Muy pocos (< 50): Puede no converger
   - Muy muchos (> 500): Tiempo desperdiciado
   - **Recomendación**: 200 suficiente para mayoría de casos

### Limitaciones

1. **Targets imposibles**: Algunos targets nunca serán top-1
   - Ejemplo: " XYZ123" (token inexistente/raro)
   - Solución: Verificar que target está en vocabulario

2. **Local minima**: Optimización puede atorarse
   - Solución: Reiniciar con diferente inicialización

3. **Memoria GPU**: Mantener gradientes consume memoria
   - Solución: Usar 4-bit quantization si necesario

---

## ✅ Checklist de Implementación

- [ ] Implementar `CISOptimizer` class
  - [ ] `__init__()` method
  - [ ] `optimize()` method
  - [ ] `_forward_with_intervention()`
  - [ ] `_compute_loss()`
  - [ ] `_backward_step()`

- [ ] Crear experimento `run_cis_optimization.py`
  - [ ] CLI arguments
  - [ ] Config loading
  - [ ] Optimizer setup
  - [ ] Results reporting

- [ ] Actualizar métricas `factual_rigidity.py`
  - [ ] `compute_geometric_cost()`
  - [ ] `compute_collateral_effects()`
  - [ ] `analyze_convergence()`

- [ ] Actualizar config `experiment.yaml`
  - [ ] CIS optimization parameters
  - [ ] Target specification

- [ ] Crear guía `CIS_OPTIMIZATION_GUIDE.md`
  - [ ] Explicación detallada
  - [ ] Ejemplos de uso
  - [ ] Interpretación de resultados

- [ ] Tests `test_cis_optimizer.py`
  - [ ] Test básico de optimización
  - [ ] Test de convergencia
  - [ ] Test de métricas

- [ ] Documentar en README
  - [ ] Nueva sección de CIS
  - [ ] Ejemplos de uso
  - [ ] Resultados esperados

---

## 🎯 Meta de Esta Fase

**Lograr esto:**

```bash
$ python src/experiments/run_cis_optimization.py --config config/experiment.yaml

# Output:
✓ Successfully flipped "Paris" → "Rome"
✓ Geometric cost: 0.189
✓ 50 optimization steps
✓ Target probability: 0.623

Next: Run on full dataset to measure average factual rigidity
```

**Con esto demostraremos:**

1. ✅ Podemos **medir cuantitativamente** la rigidez factual
2. ✅ La optimización **converge** a perturbaciones mínimas
3. ✅ Los resultados son **interpretables** y **reproducibles**
4. ✅ El sistema está **listo** para experimentos a escala

---

¿Listo para implementar? 🚀
