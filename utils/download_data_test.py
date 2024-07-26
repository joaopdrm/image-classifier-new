import os
import requests

# URLs de imagens de gatos e cachorros
cat_image_urls = [
    "https://placekitten.com/400/400",
    "https://placekitten.com/401/401",
    "https://placekitten.com/402/402",
    "https://placekitten.com/403/403",
    "https://placekitten.com/404/404",
    "https://placekitten.com/405/405",
    "https://placekitten.com/406/406",
    "https://placekitten.com/407/407",
    "https://placekitten.com/408/408",
    "https://placekitten.com/409/409",
    "https://placekitten.com/410/410",
    "https://placekitten.com/411/411",
    "https://placekitten.com/412/412",
    "https://placekitten.com/413/413",
    "https://placekitten.com/414/414",
    "https://placekitten.com/415/415",
    "https://placekitten.com/416/416",
    "https://placekitten.com/417/417",
    "https://placekitten.com/418/418",
    "https://placekitten.com/419/419"
]

dog_image_urls = [
    "https://placedog.net/400/400",
    "https://placedog.net/401/401",
    "https://placedog.net/402/402",
    "https://placedog.net/403/403",
    "https://placedog.net/404/404",
    "https://placedog.net/405/405",
    "https://placedog.net/406/406",
    "https://placedog.net/407/407",
    "https://placedog.net/408/408",
    "https://placedog.net/409/409",
    "https://placedog.net/410/410",
    "https://placedog.net/411/411",
    "https://placedog.net/412/412",
    "https://placedog.net/413/413",
    "https://placedog.net/414/414",
    "https://placedog.net/415/415",
    "https://placedog.net/416/416",
    "https://placedog.net/417/417",
    "https://placedog.net/418/418",
    "https://placedog.net/419/419"
]

# Função para baixar e salvar imagens
def download_images(image_urls, save_dir, prefix):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, url in enumerate(image_urls):
        image_response = requests.get(url)
        image_path = os.path.join(save_dir, f'{prefix}{i+1}.jpg')

        with open(image_path, 'wb') as f:
            f.write(image_response.content)

    print(f'Downloaded {len(image_urls)} images to {save_dir}')

# Diretórios para salvar as imagens
cat_dir = "../dataset/cats"
dog_dir = "../dataset/dogs"

# Baixar imagens de gatos e cachorros
download_images(cat_image_urls, cat_dir, 'cat')
download_images(dog_image_urls, dog_dir, 'dog')
