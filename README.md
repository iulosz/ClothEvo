# ClothEvo — Surreal Textile Lab + XPBD

Aplicação desktop em Rust, organizada em seis crates, para criação de moldes 2D, montagem de roupas
em 3D e simulação XPBD. O Pattern Studio inclui curvas lineares/Bézier, triangulação CDT, costuras,
piques, simetria, Cut & Sew, dobras e pregas. A física integra restrições de tecido e costura, contato
e auto-colisão entre superfícies, atrito por material e colisão corporal por SDF. Avatares estáticos
OBJ/FBX podem ser importados para visualização e colisão; o renderer WGPU é integrado ao egui.

**Estado do projeto:** protótipo em desenvolvimento. Alguns recursos são aproximados: a auto-colisão
usa um orçamento de candidatos para proteger o desempenho; linhas de dobra mapeiam para arestas
existentes da malha; e avatares importados são estáticos, sem esqueleto ou deformação animada. A
colisão por SDF é aproximada e depende da qualidade e topologia da malha. Ainda não é um substituto completo do CLO3D/Marvelous Designer nem um pipeline de produção.

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
cargo test --workspace                # testes unitários do workspace
cargo run --release -p cloth-sim --example bench   # benchmark do solver sem janela
cargo run --release -p cloth-sim --example diag -- 1 24 20 1 10   # diagnóstico de estabilidade
```

O wgpu escolhe sozinho Vulkan, DX12 ou Metal. Para forçar um: `WGPU_BACKEND=vulkan|dx12|metal|gl`.

## Interface de trabalho

A interface usa uma identidade editorial surrealista com azul-cobalto, magenta e acentos ocre. A navegação superior organiza **Ambiente 3D**, **Moldes 2D**, **Posicionamento 3D** e o menu **View**. O painel Moldes 2D identifica diretamente o editor de geometria vetorial, e o estado do avatar padrão é destacado quando o arquivo externo não está disponível. Em View, **3D + 2D** abre as duas viewports, **Visualização de malhas e costura** controla wireframe e costuras, e **Métricas** mostra os diagnósticos do solver num painel inferior redimensionável.

Uma sessão comum inicia no Ambiente 3D com chão e avatar procedural; se um arquivo padrão `Sarah.fbx` estiver num dos caminhos de recursos examinados, ele é carregado automaticamente. A cena de benchmark/demonstração não é carregada como a experiência normal; as cenas internas permanecem disponíveis para testes e captura por `CLOTHEVO_SCENE`.

## Controles

* Arrastar com o botão esquerdo: orbitar · botão direito ou do meio: mover · roda: zoom.
* Painel esquerdo: cena, resolução da malha, presets de tecido (algodão, seda, jeans, malha),
  rigidez de stretch/shear/flexão/costura, amortecimento, atrito, espessura, substeps e iterações.
* O painel **Avatar e colisão corporal** importa avatares estáticos OBJ/FBX e configura a colisão
  corporal; ela pode ser desligada sem desativar a auto-colisão do tecido.
* Todos os parâmetros podem ser alterados com a simulação rodando.

## Cenas de teste

1. **Pano pendurado** por dois cantos — stretch, shear e flexão sob grande movimento.
2. **Pano sobre esfera** — colisão com atrito.
3. **Tubo costurado** — dois painéis costurados nas laterais caindo em volta de um cilindro.

## Estrutura

```
crates/
├─ pattern-core/  modelo serializável de moldes, peças, costuras, piques, detalhes e materiais
├─ pattern-geom/  geometria 2D: curvas Bézier, comprimento de arco, validação e operações de edição
├─ pattern-mesh/  triangulação CDT dos moldes e refinamento da malha
├─ cloth-sim/     solver XPBD sem dependência de render: partículas SoA, stretch/shear/costura,
│                 flexão, contatos corpo-tecido e auto-colisão VV/VT/EE, cenas e benchmarks
├─ cloth-render/  renderer wgpu: câmera orbital, tecido de dois lados, wireframe e costuras
└─ cloth-app/     janela e interface egui: Pattern Studio, montagem 3D, avatar OBJ/FBX e métricas
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

* Na Fase 0 ainda não havia auto-colisão tecido–tecido; ela foi incorporada posteriormente e está
  descrita em **Estabilidade da simulação**.
* Na Fase 0 havia apenas colisores analíticos para o manequim procedural; a importação de avatares
  OBJ/FBX estáticos com SDF foi adicionada na Etapa 5.
* A malha inicial vinha de grades; a triangulação CDT e o refinamento de moldes reais foram
  desenvolvidos nas etapas do Pattern Studio.


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

O sistema remapeia referências de costura e entalhes em trechos de borda que continuam íntegros. Linhas internas, dobras e eixos de pregas que cruzam o corte são divididos em fragmentos aproximados e preservados em cada peça; curvas inteiramente de um lado mantêm a representação original. Pence e furos que atravessam o corte, linha de fio ambígua e costuras existentes que dependem de segmentos divididos ainda são recusados sem alterar o projeto.

As novas costuras conectam as bordas recém-criadas no grafo do projeto. A aplicação de folga diferencial (*ease*) continua sendo uma etapa separada do solver físico.

## Montagem 2D → 3D e física por molde

Cada molde possui um painel **Física e montagem 3D por molde** no Pattern Studio:

- **Material:** densidade superficial (kg/m²), rigidez no urdume e na trama (N/m), cisalhamento (N/m), flexão (N·m) e espessura de colisão (m).
- **Discretização:** a distância entre partículas é lida individualmente por molde durante a montagem da malha triangular. Valores menores preservam mais detalhes, mas aumentam o custo computacional.
- **Transformação 3D:** deslocamento XYZ em metros e rotação XYZ em graus, aplicada em torno do centro do molde. A posição inicial conserva a organização relativa do layout 2D; os offsets e rotações servem para colocar os painéis ao redor do corpo antes de simular.
- **Âncoras:** selecione um vértice do contorno e use **Fixar vértice ... na simulação** para criar um ponto preso; o vértice fica marcado em vermelho. A fixação é armazenada no projeto e mantida ao iniciar a simulação.
- **Costuras:** o comando **Montar e simular na viewport 3D** triangula cada molde com seu próprio espaçamento, aplica as transformações e materiais, configura as âncoras e converte as referências de aresta em restrições de costura XPBD.

O construtor do solver aceita `PanelMaterial` por painel. As diferenças de densidade alteram a massa por partícula; as rigidezes anisotrópicas e de flexão alteram a compliance das respectivas restrições; a espessura altera o afastamento de colisão por partícula. Os controles globais continuam funcionando como multiplicadores gerais relativos à configuração com a qual a simulação foi montada. A folga diferencial (*ease*) ajusta o comprimento de repouso das arestas selecionadas para acomodar a diferença-alvo entre as bordas.



## Avatar procedural e colisão tecido-corpo

A montagem 2D→3D utiliza um manequim procedural ajustável no Pattern Studio. As medidas são gravadas no projeto `.clothevo`: altura, largura dos ombros, cintura e quadril. Projetos antigos recebem as medidas padrão ao serem carregados.

A física e a viewport usam os mesmos 18 colisores analíticos: elipsoides para cabeça, pescoço e tronco; cápsulas para braços, antebraços, coxas e canelas; e volumes elipsoidais para mãos e pés. A projeção de colisão usa campos de distância assinada (SDF), aproveitando a espessura do tecido e o atrito existentes no solver.

**Limitações desta etapa:** é um manequim composto para colisão, não uma malha anatômica de alta resolução. Importação de avatares OBJ, FBX e glTF, colisão contra malhas arbitrárias e ajuste anatômico avançado continuam como etapas futuras.


## Sistema avançado de costura

No Pattern Studio, selecione uma costura existente para editar os moldes, intervalos de segmento, orientação, nome e diferença-alvo de comprimento. A diferença assinada `A − B` é expressa em metros: valor positivo significa que A deve ser mais longa; negativo significa que B deve ser mais longa. A diferença é aplicada ao comprimento de repouso das arestas de contorno, e não apenas salva como metadado.

O solver faz pareamento das duas bordas pelo comprimento de arco normalizado e cobre os vértices de ambos os lados mesmo quando a malha tem densidades diferentes. Piques (notches) podem ser adicionados, editados e removidos; os piques de ambas as bordas são pareados por progresso de arco e recebem vínculos explícitos no solver. O canvas 2D mostra piques e costuras, com vermelho para diferenças medidas que não coincidem com a meta. Na viewport 3D, vínculos cuja abertura supera 15 mm são realçados em vermelho; essa abertura é um indicador de diagnóstico, não uma estimativa de força em newtons.

Ações disponíveis: criar costura, selecionar e alterar uma costura existente, inverter a direção de A ou B, selecionar intervalos com vários segmentos e descosturar. A simulação atual ainda costura apenas bordas do contorno; linhas de costura internas exigem um mapeamento de malha próprio em uma etapa futura.


## Referência de colisão tecido–corpo (Etapa 1)

A suíte do `cloth-sim` inclui a queda sobre esfera analítica e uma cena reproduzível de tecido
sobre um torso fechado, consultado por SDF de malha. O torso de referência mede 1,4 × 0,8 × 0,7 m
e é gerado no código, sem depender de um arquivo de avatar.

Execute `cargo test -p cloth-sim` para validar os cenários. Para observar as métricas em execução:

```bash
cargo run --release -p cloth-sim --example diag -- 4 24 20 1 10
```

O diagnóstico informa projeções de contato, trajetórias examinadas pelo CCD aproximado, contatos
interceptados, penetração máxima e tempos do passo/restrições/colisores.
A quinta cena (`5`) reproduz um cruzamento do torso em um único substep:
`cargo run --release -p cloth-sim --example diag -- 5 24 1 1 0.1`. Ela deve registrar contatos
CCD e manter o tecido fora da superfície do torso.

## Etapa 2 — Detecção contínua aproximada tecido–corpo

O solver examina trajetórias de partículas longas entre a posição anterior e a prevista, usando
avanços limitados pelas distâncias dos SDFs. Ao encontrar a primeira superfície de contato, remove
a componente do movimento dirigida para dentro do corpo e preserva o deslizamento tangencial.
O contato corporal também é projetado após cada iteração XPBD, reduzindo a chance de as restrições
reintroduzirem penetração após uma correção.

Os contadores `body_sweep_tests` e `body_swept_contacts` mostram quantas trajetórias foram examinadas
e quantos contatos foram interceptados; a interface e o exemplo `diag` expõem esses valores.
Esta é uma forma contínua aproximada baseada em SDF, não uma implementação exata de CCD
triângulo-triângulo. Ela reduz o tunneling, mas precisa ser validada com diferentes resoluções de
SDF, tecidos e movimentos antes de ser considerada suficiente para produção.

## Estabilidade da simulação

Na viewport de simulação, ajuste **substeps por quadro** e **iterações por substep**: aumentar substeps costuma estabilizar movimentos rápidos; aumentar iterações melhora a convergência das restrições, com custo adicional. O limite de velocidade por partícula atua como salvaguarda contra deslocamentos extremos. Se forem encontrados valores não finitos, a simulação pausa e retorna ao estado inicial da cena para impedir que geometria inválida continue sendo propagada.

**Auto-colisão** continua desligada por padrão devido ao custo em malhas densas. Além dos pares
vértice-vértice, o solver discreto agora busca contatos vértice-triângulo e aresta-aresta dentro de uma
grade espacial de AABBs, atualizada a cada iteração XPBD. Os contatos respeitam as espessuras por
material, massas inversas e partículas fixadas; as projeções são seriais para evitar escritas
concorrentes. A topologia estática é montada uma vez, com arestas únicas, faces adjacentes, incidência
por vértice e exclusões locais em torno de costuras.

Os diagnósticos mostram candidatos, verificações, contatos, candidatos descartados, sobrecarga de
primitivas, penetração antes/depois e tempos separados de broadphase e resolução. A cena 6 do exemplo
`diag` sobrepõe dois painéis para exercitar contatos entre superfícies. A broadphase inclui as caixas de movimento entre posições anteriores e previstas. Para candidatos
VT/EE sem contato na posição final, o solver usa avanço conservador aproximado ao longo das trajetórias
relativas e projeta o contato encontrado; não é CCD exato de triângulos deformáveis. As cenas 6 e 7 do
exemplo `diag` exercitam, respectivamente, painéis sobrepostos e uma travessia em um substep:

```bash
cargo run --release -p cloth-sim --example diag -- 6 12 1 4 0.1
cargo run --release -p cloth-sim --example diag -- 7 12 1 1 0.1
```

A cena rápida mede cruzamentos interceptados; os resultados ainda precisam ser comparados em diferentes
resoluções, espessuras e velocidades antes de considerar a detecção suficiente para produção.

O atrito da seção **Física e montagem 3D por molde** é configurável por tecido e afeta o contato com colisores e outras partes do pano. Os controles globais de atrito e espessura continuam atuando como multiplicadores. As métricas mostram velocidade máxima, deslocamento do último substep, partículas acima do limiar de velocidade e aberturas críticas nas costuras.

Os comandos **Pausar**, **Passo único** e **Reiniciar** são independentes: o passo único pausa antes de avançar exatamente um passo fixo; reiniciar restaura o estado de colocação e mantém a simulação pausada.


## Posicionamento 3D pré-simulação

O Pattern Studio usa o arquivo PNG raster da referência **T-POSE** em `assets/t_pose_reference.png`, exibido diretamente como textura no canvas 2D pela opção **Guia anatômico T-Pose**. O contorno não é vetorizado nem reconstruído. A imagem conserva sua proporção de pixels (585 × 632) e é exibida com altura de referência de 1,72 m e largura derivada dessa proporção; é um gabarito visual, não um avatar de fabricação.

Use **Posicionar moldes na cena 3D** para gerar uma prévia dos painéis em torno do manequim sem iniciar automaticamente a física. Selecione um painel e escolha no painel lateral o modo **Mover**, **Girar**, **Escalar** ou **Transformar**:

**Layout automático (projeção frontal):** sem ponto de posicionamento, cada molde é colocado no 3D exatamente onde está no canvas 2D: o eixo X do molde vira o eixo X do mundo e o eixo Y 2D usa a mesma referência do guia T-Pose (origem 2D a 0,80 m do chão para um avatar de 1,72 m, proporcional à altura). O molde fica plano, voltado para +Z, na frente do ponto mais saliente do corpo sob a sua área, com a folga da espessura e da camada. A posição depende só do próprio molde, e não dos demais; para moldes nas costas use um ponto de posicionamento. Projetos que já tinham deslocamentos salvos no gizmo podem aparecer em outro lugar, porque eles são relativos a esse layout.

- **Mover**: arraste as setas coloridas para deslocar o molde por eixo.
- **Girar**: arraste os anéis para rotacionar o molde no eixo correspondente.
- **Escalar**: arraste os quadrados nas extremidades dos eixos; como o molde é uma superfície 2D, o controle normal ao plano aplica escala uniforme no plano.
- **Transformar**: combina setas, anéis e controles de escala numa única ferramenta.
- Arrastar o corpo do painel no modo **Mover** ainda permite posicionamento livre no plano da câmera. Arrastar fora dos painéis orbita; botão direito/meio move a câmera; a roda controla o zoom.
- Use **Restaurar transformação deste molde** para recuperar posição, rotação e escala originais e **Iniciar simulação de tecido** quando terminar.

As transformações são gravadas no projeto e no estado de repouso do solver. A rotação é aplicada sem reconstruir a malha; após escalar, a prévia é reconstruída ao soltar o manipulador para recalcular áreas, massas e restrições de repouso. Os comandos de costura continuam ligados aos vértices de contorno preservados pela triangulação CDT.


### Topologia da malha de tecido

A triangulação CDT preserva os vértices de contorno usados pelas costuras, e os pontos internos agora usam linhas alternadas com deslocamento de meia célula (amostragem triangular). Isso reduz o padrão ortogonal da grade anterior e tende a distribuir melhor os comprimentos das arestas internas. O espaçamento por molde continua configurável em **Distância entre partículas**; valores menores geram malha mais densa e elevam o custo da simulação.


### P1 — Integridade dos piques, dobras e pregas

A inserção de um vértice subdivide parametricamente os piques do segmento: os piques anteriores ao corte mantêm o segmento esquerdo, os posteriores migram para o segmento direito, e os índices seguintes são deslocados. A exclusão de uma âncora remapeia os piques para a nova aresta combinada por fração do comprimento de arco dos segmentos antigos; a exclusão da primeira âncora também trata explicitamente o fechamento do contorno. As operações de inserção e exclusão são validadas antes de confirmar a alteração no projeto.

Linhas de dobra do tipo **Vale** e **Montanha** agora mapeiam para arestas internas existentes da malha. Essas arestas recebem rigidez de flexão local e um ângulo diedral-alvo em repouso; linhas do tipo **Decorativa** permanecem apenas visuais. O ângulo inicial de uma dobra física é de 30° e a rigidez inicial é 35% da rigidez de flexão base. Arquivos antigos sem o campo de ângulo usam esse valor padrão.

Uma **prega** gera dois vincos paralelos aproximados em lados opostos do eixo, separados pela profundidade definida e com ângulos-alvo opostos. O mapeamento usa caminhos no grafo de adjacência da malha existente: não insere vértices nem modifica a triangulação CDT. O painel de montagem informa quantas arestas de dobra/prega foram mapeadas; se nenhuma aresta interna puder ser encontrada, é preciso aumentar a resolução do molde ou revisar o traçado.

### P2 — Integridade dos detalhes no Cut & Sew

Ao executar um corte, o editor mantém linhas internas, dobras e eixos de pregas. Se um desses elementos cruza o caminho de corte, a curva é particionada por lado e convertida em fragmentos poligonais, com o encontro calculado por busca binária entre amostras. Curvas inteiramente contidas em uma das peças mantêm a representação original, incluindo segmentos Bézier. A aproximação usa 64 intervalos por segmento original; isso evita perder o detalhe, mas fragmentos cruzados não são uma subdivisão Bézier exata.

A operação continua transacional: referências de costura existentes, pences ou furos que não podem ser reparticionados com segurança causam cancelamento sem mutar o projeto. Linhas de fio e linhas de costura ambíguas também continuam protegidas em vez de serem atribuídas arbitrariamente.

### P3 — Validação preventiva dos dados de molde

A validação do projeto verifica valores finitos não só no contorno, mas também em linhas de costura e internas, dobras, eixos de pregas, linha de fio, pences, furos, anotações e coeficientes de graduação. A escala de montagem precisa ser finita e positiva. Como salvar e carregar passam por PatternProject::validate, dados inválidos são rejeitados antes de serem persistidos ou usados nas próximas etapas.

Os testes incluem regressões para geometria não finita, escalas inválidas, coeficientes de graduação e fragmentação de linhas cruzadas pelo Cut & Sew. P3 pode continuar com avisos de qualidade geométrica (por exemplo, autointerseções, contornos degenerados e margens de costura sem solução) sem transformar esses avisos em rejeições automáticas antes de haver regras geométricas robustas para esses casos.


### Limites de segurança da simulação e da malha

A auto-colisão examina até 96 candidatos por partícula, com um orçamento global de 2 milhões de pares por substep e sem pré-alocação proporcional à malha. Isso evita crescimento quadrático de memória quando vértices colapsam numa célula. Em congestionamentos extremos, alguns pares deixam de ser examinados; o limite é uma proteção de estabilidade/desempenho, não uma garantia de colisão exaustiva.

A preparação da malha recusa contornos com mais de 4.096 vértices, grades acima de 2 milhões de candidatos, custo estimado acima de 25 milhões de verificações ponto-contorno ou malhas acima de 500 mil vértices. Quando um limite é ultrapassado, aumente o espaçamento entre partículas ou simplifique o contorno.

Transformações 3D importadas são limitadas a translações entre -100 m e 100 m, rotações até ±360.000 graus e escala entre (0, 100]. Valores fora desses limites são rejeitados antes de serem convertidos para o solver em f32.


## Etapa 4 — contato, materiais e costuras no mesmo ciclo

A simulação resolve as restrições de estiramento, cisalhamento, flexão e costura e, em cada iteração, atualiza os contatos entre camadas e com o avatar. A montagem usa o campo `layer` para posicionar os moldes; durante a simulação, essa camada não desativa colisões nem aplica uma força de afastamento permanente.

A espessura efetiva por superfície é a espessura global multiplicada pelo fator do material. Para dois tecidos, o solver usa metade da soma das espessuras, com uma pequena margem numérica aplicada uma única vez. No avatar, o tecido usa sua espessura como afastamento da superfície do colisor. As rotinas vértice–vértice, vértice–triângulo e aresta–aresta compartilham a regra entre tecidos.

O atrito normal e tangencial é resolvido separadamente. O atrito estático tenta remover o movimento tangencial relativo; ao exceder seu limite de Coulomb, a correção passa ao limite dinâmico. O coeficiente entre materiais é a média dos dois coeficientes por superfície multiplicada pelo controle global. O solver limita o coeficiente dinâmico ao estático. A resposta normal acumulada é uma medida posicional XPBD usada como suporte do limite de atrito, não uma força calibrada. O histórico do atrito é descartado quando o contato desaparece.

Projetos antigos mantêm o campo existente como coeficiente estático e recebem `0,32` como padrão para o novo coeficiente dinâmico. O painel de propriedades permite ajustar ambos; os diagnósticos da simulação mostram contatos estáticos e deslizantes, velocidade tangencial relativa e a maior resposta normal posicional.

As regressões do solver verificam os limites estático e dinâmico, o deslizamento vértice–triângulo, o atrito tecido–avatar e a remoção do histórico após a separação. As cenas de camadas, costuras e cruzamento rápido continuam exercitando o ciclo de restrições e contatos.


## Etapa 5 — avatares estáticos OBJ/FBX

No fluxo de uso, selecione OBJ ou FBX no painel e ajuste a altura, o giro em Y e a resolução
do campo de colisão. Use a exibição sólida ou wireframe para conferir o modelo. A colisão corporal
pode ser desligada independentemente da auto-colisão do tecido.

O painel **Avatar e colisão corporal** importa arquivos OBJ e FBX fora da thread da interface. O
FBX converte eixos/unidades para Y-up e metros; para OBJ, o importador estima o eixo vertical. A
geometria é centralizada, apoiada no chão, ajustada à altura configurada e pode receber um giro em
Y. A malha renderizada e o SDF de colisão usam a mesma transformação.

A malha visual e a física são representações separadas: a viewport pode mostrar o modelo sólido ou
seu wireframe, enquanto o solver consulta um campo de distância voxelizado. A BVH acelera a
construção do campo; as consultas do solver usam interpolação trilinear no SDF. O espaçamento do
campo é ajustável de 4 a 30 mm e pode aumentar automaticamente até 80 mm para manter a grade abaixo
de 2,5 milhões de células. O painel informa dimensões, contagem de vértices/triângulos, resolução,
tempo de importação e tempo de construção do campo.

Malhas fechadas com arestas manifold e orientação consistente recebem classificação assinada de
interior/exterior. Malhas abertas, não-manifold ou com winding inconsistente são mantidas para
contato de proximidade, mas usam distância sem sinal; a interface exibe o diagnóstico porque esse
modo não identifica de forma confiável qual lado é o interior. Se a geração do SDF falhar, o modelo
continua visível e a colisão volta ao manequim procedural. O usuário pode desligar a colisão do
avatar sem desligar a auto-colisão do tecido.

Os contatos usam a mesma resolução corporal do solver XPBD, incluindo projeção pelo gradiente do
campo, atrito e o detector contínuo aproximado já usado para trajetórias rápidas. As métricas por
passo incluem projeções, verificações e contatos CCD, maior penetração observada e tempo de colisão.
O espaçamento do SDF é um compromisso entre detalhe, memória e preparação; nenhum limite universal
de tempo é assumido para modelos diferentes.

A etapa cobre avatares estáticos. Rigging, animação esquelética, atualização de colisão para malha
deformável e simplificação automática validada por distância permanecem fora do escopo.