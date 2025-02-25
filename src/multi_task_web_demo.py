import gradio as gr
from utils.zero_shot_classification import classify_zero_shot, load_zero_shot_model
from src.multi_task_pipeline import multi_task_pipeline, load_category_model

def toggle_custom_labels(use_zero_shot):
    return gr.update(visible=use_zero_shot)

# Modelleri yükle
category_model, tokenizer, labels = load_category_model()
zero_shot_model, zero_shot_tokenizer = load_zero_shot_model()

# Gradio Arayüzü
with gr.Blocks() as iface:
    gr.Markdown("## Multi-Task Haber Analizi")
    gr.Markdown(
        "Girilen haber metnini analiz ederek kategori, anahtar kelime, tarih/lokasyon ve özetleme işlemleri yapar. Opsiyonel olarak Zero-Shot sınıflandırma kullanılabilir.")

    text_input = gr.Textbox(lines=5, placeholder="Haber metninizi buraya yazın...", label="Haber Metni")
    zero_shot_checkbox = gr.Checkbox(label="Zero-Shot Sınıflandırmayı Kullan")
    custom_labels_input = gr.Textbox(lines=1, label="Etiketler", placeholder="Zero-Shot Etiketlerini Virgülle Ayırın (Opsiyonel)",
                                     visible=False)

    zero_shot_checkbox.change(toggle_custom_labels, inputs=zero_shot_checkbox, outputs=custom_labels_input)

    output = gr.JSON()
    with gr.Row():
        submit_button = gr.Button("Analiz Et")

    submit_button.click(lambda text, use_zero_shot, custom_labels: (
                        multi_task_pipeline(text, category_model, tokenizer, labels, use_zero_shot, zero_shot_model, zero_shot_tokenizer, custom_labels)),
                        inputs=[text_input, zero_shot_checkbox, custom_labels_input],
                        outputs=[output])

# Uygulamayı başlat
if __name__ == "__main__":
    iface.launch(share=True, show_api=True)
