''''Este é um leitor de csv para dados de IMC (ou BMI).
Ele recebe como entrada um csv com dados de ID, height(altura), weight(peso) e retorna um relatório contendo:
*média de altura
*média de peso
*média de IMC
*maior altura
*menor altura
*maior peso
*menor peso

Para a entrada, o csv deve estar no formato:
primeira linha: nomes (labels)
próximas linhas: valores numéricos'''

linhas = []
alturas = []
pesos = []
lista_imc = []


print("Digite o CSV (pressione ENTER vazio para finalizar):")

#lê o csv
while True:
    linha = input()
    if linha == "":
        break
    linhas.append(linha)

cabecalho = linhas[0].split(",")

#processa os dados
for linha in linhas[1:]:
    dados = linha.split(",")

    altura = float(dados[1])
    peso = float(dados[2])

    imc = peso / (altura ** 2)

    lista_imc.append(imc)
    alturas.append(float(dados[1])) 
    pesos.append(float(dados[2]))

media_alturas = sum(alturas) / len(alturas)
media_pesos = sum(pesos) / len(pesos)
media_imc = sum(lista_imc) / len(lista_imc)

maior_altura = max(alturas)
menor_altura = min(alturas)
maior_peso = max(pesos)
menor_peso = min(pesos)

print("Impressão do relatório: \n")
print(f"média dos altura: {round(media_alturas, 2)}")
print(f"média dos pesos: {round(media_pesos, 2)}")
print(f"média dos imc: {round(media_imc, 2)}")
print(f"maior altura: {maior_altura}")
print(f"menor altura: {menor_altura}")
    

