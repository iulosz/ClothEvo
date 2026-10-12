# ClothEvo — Fase 0

Protótipo para validar os maiores riscos técnicos de um programa de moldes + simulação de tecido
(estilo CLO3D / Marvelous Designer): **estabilidade e desempenho do solver XPBD**, colisão com
primitivas e **costuras 2D → 3D**. Ainda não há editor 2D, avatar glTF nem DXF — isso vem nas
próximas fases.

## Como rodar

1. Instale o Rust (inclui o `cargo`): <https://rustup.rs>
   - **Windows:** baixe o `rustup-init.exe`. Se pedir, instale também o *Visual Studio Build Tools*
     (carga "Desenvolvimento para desktop com C++").
   - **Linux:** `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
     e, no Ubuntu/Debian, `sudo apt install build-essential pkg-config libx11-dev libxkbcommon-dev libwayland-dev libvulkan1`.
   - **macOS:** `xcode-select --install` e depois o comando do rustup acima.
2. Na pasta do projeto:

```bash
cargo run --release -p cloth-app      # abre o ClothEvo (use --release: o solver é pesado)
cargo test --workspace                # 22 testes
cargo run --release -p cloth-sim --example bench   # benchmark do solver sem janela
cargo run --release -p cloth-sim --example diag -- 1 24 20 1 10   # diagnóstico de estabilidade
```

O wgpu escolhe sozinho Vulkan, DX12 ou Metal. Para forçar um: `WGPU_BACKEND=vulkan|dx12|metal|gl`.

## Controles

* Arrastar com o botão esquerdo: orbitar · botão direito ou do meio: mover · roda: zoom.
* Painel esquerdo: cena, resolução da malha, presets de tecido (algodão, seda, jeans, malha),
  rigidez de stretch/shear/flexão/costura, amortecimento, atrito, espessura, substeps e iterações.
* Todos os parâmetros podem ser alterados com a simulação rodando.

## Cenas de teste

1. **Pano pendurado** por dois cantos — stretch, shear e flexão sob grande movimento.
2. **Pano sobre esfera** — colisão com atrito.
3. **Tubo costurado** — dois painéis costurados nas laterais caindo em volta de um cilindro.

## Estrutura

```
crates/
├─ cloth-sim/     solver XPBD (sem dependência de render): partículas SoA, restrições de
│                 distância (stretch/shear/costura) e flexão (ângulo diedral), coloração de
│                 grafo + rayon, colisores SDF (plano, esfera, cilindro), cenas e benchmarks
├─ cloth-render/  renderer wgpu próprio (shader WGSL, câmera orbital, malha de tecido de dois
│                 lados, wireframe, costuras), integrado ao egui por paint callback
└─ cloth-app/     janela (eframe) + painel de parâmetros + métricas
```

Decisões de projeto que valem para as próximas fases:

* **Comprimentos de repouso vêm do molde 2D** (`rest_uv`), não da posição 3D inicial.
* **Rigidez física independente de resolução**: stretch/shear em N/m, flexão em N·m.
* O solver não depende de nenhuma API gráfica, para poder migrar para compute shader.

## Resultados da validação (Fase 0)

Medidos num ambiente de 2 vCPUs, 20 substeps por frame de 1/60 s, release:

| Partículas | 1 thread | 2 threads |
|-----------:|---------:|----------:|
| 2.401      | 10,0 ms  | 14,5 ms   |
| 7.225      | 30,6 ms  | 24,5 ms   |
| 14.641     | 63,2 ms  | 44,4 ms   |
| 28.900     | 124,7 ms | 78,0 ms   |

* O solver é **estável** (sem NaN, deformação máxima de poucos % nas cenas de teste) e os
  15 testes do `cloth-sim` cobrem o gradiente da flexão por diferenças finitas, a coloração,
  a colisão, as costuras e a igualdade entre execução serial e paralela.
* **Risco confirmado:** em CPU, com 20 substeps, só malhas de ~2–5 mil partículas cabem em
  16,7 ms (60 Hz) nessa máquina. Malhas de roupa reais (20 mil+) pedem **solver em GPU** (compute
  shader) ou menos substeps com melhores preditores. Em malhas pequenas o rayon é mais lento que
  o modo serial (overhead por cor de restrição).

## Limitações conhecidas da Fase 0

* Sem auto-colisão tecido–tecido.
* Colisores analíticos apenas; avatar com BVH/SDF entra na Fase 4.
* A malha vem de grades; a triangulação CDT de moldes reais entra na Fase 1/2.


## Fase 1 — base do Pattern Studio

A primeira tranche do editor de moldes cria três crates independentes:

- pattern-core: modelo serializável do projeto, peças, contorno de corte, linha de costura,
  margem de costura, linhas internas, pences, entalhes, furos, linha de fio, anotações e costuras.
- pattern-geom: amostragem de Bézier, comprimento de arco, validação de contorno,
  ponto-no-polígono e offset poligonal com junção miter.
- pattern-mesh: triangulação Delaunay restrita (CDT), refinamento simples por pontos internos
  e preservação dos índices de borda para futura correspondência de costura.

Os três módulos não dependem do app ou do renderer. As medidas geométricas internas são em metros;
a unidade escolhida no projeto é informação de apresentação/conversão. O offset atual trata
polígonos simples e pode rejeitar casos degenerados ou offsets que se autointersectem; operações
booleanas robustas, furos na triangulação e margem variável por aresta continuam como trabalho futuro.

Comandos de validação após obter dependências Rust:
cargo test -p pattern-core -p pattern-geom -p pattern-mesh


## Pattern Studio 0.2 — edição do canvas

O editor 2D agora oferece dois modos: **Desenhar** (adicionar pontos ao rascunho) e **Editar**
(selecionar/arrastar vértices e alças Bézier). No modo Editar, clicar perto de uma borda seleciona
o segmento; **Inserir vértice** subdivide linhas ou curvas cúbicas, preservando a forma Bézier por
De Casteljau e atualizando as referências de intervalos de costura.

A viewport 2D aceita zoom pela roda do mouse ancorado no cursor e pan com botão direito ou do meio.
A opção **Ajustar à grade (1 cm)** aplica snap ao desenhar e mover pontos. **Excluir vértice**
funciona em contornos poligonais lineares com pelo menos quatro vértices; contornos Bézier precisam
ser editados por subdivisão/alças, para evitar alterações silenciosas de forma. Exclusões que
tornariam um intervalo de costura descontínuo são recusadas sem modificar o projeto.

O editor ainda é uma base de produção inicial: ferramentas de seleção de peça inteira, undo/redo,
edição de margem por aresta e janelas de diálogo para arquivos estão planejadas para etapas futuras.


## ClothEvo 0.3 — histórico, seleção de peças e arquivos

O Pattern Studio agora oferece histórico de edições Undo/Redo (atalhos Ctrl+Z, Ctrl+Y e Ctrl+Shift+Z),
com limite de 100 estados, incluindo pontos de rascunho e operações em moldes. Peças podem ser
selecionadas clicando no interior do contorno, movimentadas no canvas e duplicadas com deslocamento
inicial de 20 cm. A duplicação não cria costuras automaticamente, evitando vínculos incorretos.

Os projetos podem ser criados, abertos e salvos por diálogos nativos. O primeiro salvamento solicita
um caminho; depois, **Salvar** atualiza esse arquivo, enquanto **Salvar como** permite escolher outro.
Os formatos .clothevo e .json usam o esquema JSON do projeto. A interface indica quando há alterações
não salvas. A opção **Enquadrar todos os moldes** calcula zoom e deslocamento para ajustar as peças à
viewport.

Limitações desta etapa: o histórico cobre as edições de moldes e rascunhos, mas não restaura o caminho
de arquivo nem os parâmetros transitórios do painel de costura; mover uma peça desloca também linhas
internas, pences, furos, linha de fio e anotações. Costuras entre peças não são duplicadas junto com
uma cópia porque é necessário definir explicitamente a nova correspondência.
