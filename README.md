# Sistema de Login com Reconhecimento Facial

Um sistema de login seguro e eficiente utilizando reconhecimento facial, desenvolvido em Python. O sistema permite cadastrar usuários através de suas características faciais e realizar login de forma rápida e segura.

## Características

- Reconhecimento facial em tempo real
- Sistema de cadastro de novos usuários
- Interface amigável com feedback visual
- Armazenamento eficiente de dados faciais
- Processamento otimizado para melhor performance
- Suporte a múltiplos usuários

## Requisitos

- Python 3.7+
- Webcam
- Bibliotecas Python:
  - OpenCV (cv2)
  - dlib
  - NumPy
  - face_recognition

## Instalação

1. Clone o repositório:
```bash
git clone [URL_DO_REPOSITORIO]
cd [NOME_DO_DIRETORIO]
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Baixe os modelos necessários:
- shape_predictor_68_face_landmarks.dat
- dlib_face_recognition_resnet_model_v1.dat

## Estrutura do Projeto

```
projeto-reconhecimento-facial/
├── main.py              # Arquivo principal do sistema
├── requirements.txt     # Dependências do projeto
├── know_faces/         # Diretório para armazenar faces conhecidas
├── face_encodings.json # Arquivo JSON com encodings faciais
└── README.md           # Este arquivo
```

## Como Usar

1. Execute o programa:
```bash
python main.py
```

2. Comandos disponíveis:
- Pressione 'c' para cadastrar um novo usuário
- Pressione 'q' para sair do sistema

3. Fluxo de cadastro:
   - Digite o nome do novo usuário
   - Posicione o rosto no centro da tela
   - Aguarde a captura dos encodings faciais
   - Pressione 'q' para finalizar o cadastro

4. Fluxo de login:
   - O sistema reconhecerá automaticamente faces cadastradas
   - Exibirá uma mensagem de boas-vindas quando reconhecer um usuário
   - Sugerirá cadastro para faces não reconhecidas

## Funcionamento Técnico

### Detecção Facial
- Utiliza o detector de faces do dlib
- Processa frames em uma thread separada para melhor performance
- Reduz a escala dos frames para otimizar o processamento

### Reconhecimento Facial
- Utiliza o modelo ResNet do dlib para extração de características faciais
- Compara características usando distância euclidiana
- Mantém múltiplos encodings por usuário para maior precisão

### Armazenamento
- Encodings faciais são salvos em formato JSON
- Estrutura eficiente para carregamento rápido
- Suporte a múltiplos encodings por usuário

## Otimizações

1. Performance:
   - Processamento em thread separada
   - Redução de escala dos frames
   - Buffer otimizado da webcam

2. Precisão:
   - Múltiplos encodings por usuário
   - Modelo ResNet para extração de características
   - Threshold de confiança ajustável

## Contribuindo

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
