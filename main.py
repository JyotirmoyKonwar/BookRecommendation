import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

df = pd.read_csv('dataset/full_emo_books.csv')
df["large_thumbnail"] = df["thumbnail"] + "&fife=w800"
df["large_thumbnail"] = np.where(
    df["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    df["large_thumbnail"],
)

raw_documents = TextLoader("dataset/tagged_descriptions.txt").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db_books = Chroma.from_documents(documents, embeddings)


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_n: int = 50,
        final_n: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_n)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_rec = df[df["isbn13"].isin(books_list)].head(initial_n)

    if category != "All":
        book_rec = book_rec[book_rec["simple_categories"] == category].head(final_n)
    else:
        book_rec = book_rec.head(final_n)

    if tone == "Happy":
        book_rec.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Suspense":
        book_rec.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_rec.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Horror":
        book_rec.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sorrow":
        book_rec.sort_values(by="sadness", ascending=False, inplace=True)

    return book_rec


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(df["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Horror", "Sorrow"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Zero shot book recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "What kind of book would you like today?",
                                placeholder = "e.g., Horror, Sherlock Holms like?")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an genre:", value = "All")
        submit_button = gr.Button("Find Recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)


if __name__ == "__main__":
    dashboard.launch()