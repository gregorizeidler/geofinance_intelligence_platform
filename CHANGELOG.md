# Changelog

## [2.0.0] - 2025-08-25

### Added
- 🛰️ **Satellite Data Integration**
  - Adicionado suporte para dados do Landsat e Sentinel
  - Novas features de densidade urbana e mudanças no uso do solo
  - Integração com Google Earth Engine para processamento de imagens
  - Análise de vegetação e áreas construídas

- 📱 **Dados de Mobilidade**
  - Nova fonte de dados de padrões de movimento
  - Features de densidade de atividade por período do dia
  - Análise de diversidade de usuários
  - Padrões de movimento temporal

### Features Adicionadas
- **Satellite Features:**
  - urban_density: Índice de densidade urbana
  - vegetation_index: Índice de vegetação
  - built_up_index: Índice de área construída
  - vegetation_density: Densidade de vegetação
  - water_index: Índice de água
  - urban_change: Detecção de mudanças urbanas

- **Mobile Features:**
  - activity_density: Densidade de atividade
  - user_diversity: Diversidade de usuários
  - morning_activity: Atividade no período da manhã
  - midday_activity: Atividade no meio do dia
  - afternoon_activity: Atividade à tarde
  - evening_activity: Atividade à noite
  - night_activity: Atividade noturna

### Melhorias
- Atualização do sistema de configuração para suportar novas fontes de dados
- Otimização do pipeline de processamento para grandes volumes de dados
- Melhor integração com sistemas de cache para features complexas
- Documentação expandida para novas fontes de dados

### Requisitos
- Novas dependências:
  - earthengine-api>=0.1.0
  - sentinelsat>=1.1.0
  - rasterio>=1.3.0

### Configuração
- Necessário configurar credenciais do Google Earth Engine
- Novos diretórios de cache para dados de satélite e móveis

### Próximos Passos
- Implementar análise temporal de mudanças urbanas
- Adicionar mais índices de sensoriamento remoto
- Expandir análise de padrões de mobilidade
- Desenvolver visualizações específicas para novos dados
