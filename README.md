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

Esta base foi expandida nas versões seguintes com seleção de peças, histórico de edição, operações
topológicas de contorno, linhas internas, simetria vinculada e Cut & Sew. A seção Pattern Studio 0.4
abaixo descreve o fluxo atual e seus limites deliberados para preservar a integridade dos projetos.


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


### Atalhos e ações do Pattern Studio 0.3

- **Ctrl+Z** desfaz; **Ctrl+Y** ou **Ctrl+Shift+Z** refaz.
- No modo **Selecionar / editar**, clique dentro de uma peça para selecioná-la e arraste para movê-la.
- Use **Duplicar peça selecionada** para criar uma cópia deslocada; as costuras não são clonadas.
- Clique em uma borda (destacada em amarelo) e use **Usar borda selecionada como A/B** para configurar uma costura visualmente.
- Abrir um projeto com alterações não salvas mostra uma confirmação antes de descartá-las. O histórico é reiniciado quando outro projeto é aberto ou criado, evitando que o caminho de arquivo mude ao desfazer edições antigas.

## Pattern Studio 0.4 — geometria vetorial, simetria e Cut & Sew

### Geometria vetorial

- **Adicionar retângulo** e **Adicionar polígono regular** criam moldes editáveis; o controle de lados vai de 3 a 12.
- A camada `pattern-geom` expõe primitivas, translação, reflexão vertical, projeção de pontos, extração reversível de intervalos, reamostragem e inserção de vértices em Bézier por De Casteljau.
- O contorno mantém segmentos lineares e Bézier cúbicos. As operações geométricas são independentes do canvas e podem ser reutilizadas por importadores, ferramentas de traçado e futuras operações booleanas.
- O modelo persistente inclui propriedades do tecido por molde, linha de fio, linhas internas, linhas de dobra, pregas, pences, entalhes, furos e anotações. Linhas, dobras, pregas e linha de fio têm ferramentas próprias no canvas.

### Simetria vinculada

1. Selecione um molde e ajuste **Eixo X (m)**. **Usar X mínimo do molde** define o eixo a partir da menor coordenada X do contorno.
2. Clique em **Criar par espelhado**. O projeto mantém uma relação de simetria serializada entre as duas peças.
3. Arrastar a peça, mover vértices/alças, converter segmentos em Bézier, inserir/excluir vértices e adicionar linhas internas, dobras, pregas, linha de fio ou alterar a margem sincroniza o molde correspondente.
4. **Desvincular par** mantém ambas as peças, mas transforma-as em peças independentes.

A simetria mantém a correspondência de índices nas operações topológicas suportadas. Costuras existentes são espelhadas quando as duas peças de suas extremidades possuem pares vinculados. As relações são validadas ao salvar/abrir e fazem parte do histórico de desfazer/refazer.

### Cut & Sew

1. Selecione o molde de origem e use o modo **Desenhar**.
2. Clique para desenhar o caminho de corte pelo interior do molde. A primeira e a última posições devem encostar no contorno; a tolerância de encaixe é de 3,5 cm.
3. Clique em **Cut & Sew**. O ClothEvo valida o caminho, divide o contorno em dois, preserva os segmentos Bézier, cria as duas peças e adiciona automaticamente uma costura entre as novas bordas de corte.
4. O vínculo de simetria é preservado: se a peça tem um par, o caminho é refletido e o corte é executado nos dois moldes na mesma operação. As duas novas peças também formam um novo par vinculado.

O sistema remapeia referências de costura e entalhes em trechos de borda que continuam íntegros. Se uma costura existente usa um segmento dividido pelo corte, ou se um detalhe interno, pence, furo, linha de fio ou prega atravessa a nova divisão, a operação é cancelada sem alterar o projeto. Nesse caso, reposicione/refaça o detalhe ou a costura e tente novamente. O corte não divide automaticamente esses detalhes que atravessam a linha.

As novas costuras conectam as bordas recém-criadas no grafo do projeto. A aplicação de folga diferencial (*ease*) continua sendo uma etapa separada do solver físico.

## Montagem 2D → 3D e física por molde

Cada molde possui um painel **Física e montagem 3D por molde** no Pattern Studio:

- **Material:** densidade superficial (kg/m²), rigidez no urdume e na trama (N/m), cisalhamento (N/m), flexão (N·m) e espessura de colisão (m).
- **Discretização:** a distância entre partículas é lida individualmente por molde durante a montagem da malha triangular. Valores menores preservam mais detalhes, mas aumentam o custo computacional.
- **Transformação 3D:** deslocamento XYZ em metros e rotação XYZ em graus, aplicada em torno do centro do molde. A posição inicial conserva a organização relativa do layout 2D; os offsets e rotações servem para colocar os painéis ao redor do corpo antes de simular.
- **Âncoras:** selecione um vértice do contorno e use **Fixar vértice ... na simulação** para criar um ponto preso; o vértice fica marcado em vermelho. A fixação é armazenada no projeto e mantida ao iniciar a simulação.
- **Costuras:** o comando **Montar e simular na viewport 3D** triangula cada molde com seu próprio espaçamento, aplica as transformações e materiais, configura as âncoras e converte as referências de aresta em restrições de costura XPBD.

O construtor do solver aceita `PanelMaterial` por painel. As diferenças de densidade alteram a massa por partícula; as rigidezes anisotrópicas e de flexão alteram a compliance das respectivas restrições; a espessura altera o afastamento de colisão por partícula. Os controles globais continuam funcionando como multiplicadores gerais relativos à configuração com a qual a simulação foi montada. A folga diferencial (*ease*) entre costuras é preservada nos dados, mas ainda não modifica o comprimento de repouso da costura física.



## Avatar procedural e colisão tecido-corpo

A montagem 2D→3D utiliza um manequim procedural ajustável no Pattern Studio. As medidas são gravadas no projeto `.clothevo`: altura, largura dos ombros, cintura e quadril. Projetos antigos recebem as medidas padrão ao serem carregados.

A física e a viewport usam os mesmos 18 colisores analíticos: elipsoides para cabeça, pescoço e tronco; cápsulas para braços, antebraços, coxas e canelas; e volumes elipsoidais para mãos e pés. A projeção de colisão usa campos de distância assinada (SDF), aproveitando a espessura do tecido e o atrito existentes no solver.

**Limitações desta etapa:** é um manequim composto para colisão, não uma malha anatômica de alta resolução. Importação de avatares OBJ, FBX e glTF, colisão contra malhas arbitrárias e ajuste anatômico avançado continuam como etapas futuras.


## Sistema avançado de costura

No Pattern Studio, selecione uma costura existente para editar os moldes, intervalos de segmento, orientação, nome e diferença-alvo de comprimento. A diferença assinada `A − B` é expressa em metros: valor positivo significa que A deve ser mais longa; negativo significa que B deve ser mais longa. A diferença é aplicada ao comprimento de repouso das arestas de contorno, e não apenas salva como metadado.

O solver faz pareamento das duas bordas pelo comprimento de arco normalizado e cobre os vértices de ambos os lados mesmo quando a malha tem densidades diferentes. Piques (notches) podem ser adicionados, editados e removidos; os piques de ambas as bordas são pareados por progresso de arco e recebem vínculos explícitos no solver. O canvas 2D mostra piques e costuras, com vermelho para diferenças medidas que não coincidem com a meta. Na viewport 3D, vínculos cuja abertura supera 15 mm são realçados em vermelho; essa abertura é um indicador de diagnóstico, não uma estimativa de força em newtons.

Ações disponíveis: criar costura, selecionar e alterar uma costura existente, inverter a direção de A ou B, selecionar intervalos com vários segmentos e descosturar. A simulação atual ainda costura apenas bordas do contorno; linhas de costura internas exigem um mapeamento de malha próprio em uma etapa futura.


## Estabilidade e auto-colisão da simulação

O painel **Solver e estabilidade** permite controlar substeps por quadro, iterações por substep, auto-colisão e velocidade máxima das partículas. O preset **Equilibrado** privilegia desempenho; **Mais estável** aumenta substeps e iterações e reduz o limite de velocidade. Pausar interrompe o avanço automático; **Avançar 1 quadro** executa exatamente um passo fixo e deixa a simulação pausada; **Reiniciar** restaura a colocação inicial e também pausa a reprodução.

A auto-colisão é opcional por custo de CPU. Ela usa uma grade espacial para encontrar partículas próximas, resolve a separação por projeção posicional e ignora pares adjacentes no triângulo ou explicitamente costurados. Isso reduz interpenetrações entre camadas de tecido, mas é uma aproximação de contato partícula–partícula: não substitui detecção contínua entre triângulos e pode deixar atravessamentos em malhas muito grossas ou com velocidades altas.

A espessura de contato já é aplicada por molde; agora o atrito também é configurável no material de cada molde. O controle global de atrito permanece como multiplicador mestre. A velocidade máxima funciona como limitador numérico para conter explosões e registrar quantas correções ocorreram no último passo.

O painel de métricas mostra velocidade máxima, partículas próximas do limite, correções de estabilidade, contatos de auto-colisão e a maior penetração estimada no último passo. Para costuras, a abertura acima do limite visual continua sendo um indicador geométrico, não uma medição calibrada de força em newtons.
