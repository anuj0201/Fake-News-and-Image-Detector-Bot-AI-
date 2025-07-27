import logging
import aiohttp
import requests
import base64
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram import Update
from sentence_transformers import SentenceTransformer, util
SIMILARITY_THRESHOLD = 0.65  # Can be tuned
# ➡️ Logging Configuration
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')
# ➡️ API Keys
GOOGLE_SEARCH_API_KEY = ""            """use ur own api keys from google cloud console"""
GOOGLE_SEARCH_CX = ""
NEWS_DATA_API_KEY = ""
VISION_API_KEY = ""

# ➡️ News Sources
NEWS_SITES = (
    "site:news.google.com OR site:bbc.com OR site:cnn.com OR site:reuters.com OR "
    "site:nytimes.com OR site:theguardian.com OR site:timesofindia.indiatimes.com OR "
    "site:ndtv.com OR site:hindustantimes.com OR site:indianexpress.com OR "
    "site:livemint.com OR site:financialexpress.com OR site:deccanherald.com OR site:thehindu.com OR "
    "site:scroll.in OR site:business-standard.com OR site:theprint.in OR site:news18.com OR "
    "site:firstpost.com OR site:swarajyamag.com OR site:thequint.com OR site:amarujala.com OR "
    "site:patrika.com OR site:zeenews.india.com OR site:timesnownews.com OR site:indiatvnews.com OR "
    "site:ians.in OR site:uniindia.com OR site:aajtak.in OR site:india24news.in OR site:ptinews.com OR "
    "site:ani.in"
)

# ➡️ Image Detection Thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.85
CONFIDENCE_VARIATION_THRESHOLD = 0.1
MIN_TERM_COUNT = 8

# ➡️ Define Sentiment Terms
POSITIVE_TERMS = ["good", "positive", "true", "correct", "authentic", "confirmed"]
NEGATIVE_TERMS = ["bad", "false", "fake", "incorrect", "misleading", "error"]

# ✅ Verify News Statement
# ✅ Verify News Statement with Sentence-Transformers
async def analyze_news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text
    query_embedding = model.encode(query, convert_to_tensor=True)

    if not query:
        await update.message.reply_text("⚠️ Please provide a valid news statement.")
        return
    
    url = f"https://www.googleapis.com/customsearch/v1?q={query} site:news.google.com OR site:bbc.com&cx={GOOGLE_SEARCH_CX}&key={GOOGLE_SEARCH_API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if "items" in data and len(data["items"]) >= 3:
            snippets = [item.get('snippet', '').lower() for item in data["items"][:5]]
            sources = "\n".join([f"[{item['title']}]({item['link']})" for item in data["items"][:5]])

            # Use Sentence-Transformer to calculate text similarity and detect similarity
            query_embedding = model.encode(query, convert_to_tensor=True)
            snippet_embeddings = [model.encode(snippet, convert_to_tensor=True) for snippet in snippets]

            # Compare query with each snippet
            similarities = [util.pytorch_cos_sim(query_embedding, snippet_embedding).item() for snippet_embedding in snippet_embeddings]
            max_similarity = max(similarities)
            
            # Sentiment analysis based on the similarity and positive/negative terms
            positive_count = sum(1 for snippet in snippets if any(term in snippet for term in POSITIVE_TERMS))
            negative_count = sum(1 for snippet in snippets if any(term in snippet for term in NEGATIVE_TERMS))

            if positive_count > negative_count:
                sentiment = "🙂 Positive"
            elif negative_count > positive_count:
                sentiment = "😡 Negative"
            else:
                sentiment = "😐 Neutral"

            # Determine Truthfulness based on similarity score and positive/negative count
            if max_similarity > 0.40 and  positive_count > negative_count and max_similarity < SIMILARITY_THRESHOLD :
                result = "✅True"
                
            elif max_similarity < 0.40 or negative_count > positive_count:
                result = "❌False"
            else:
                result = "✅True"
            response_text = (
                f"**🔎 News Analysis:**\n\n"
                f"**Result:** {result}\n"
                f"**Sentiment:** {sentiment}\n"
                f"**Max Similarity with Snippets:** {max_similarity:.2f}\n\n"
                f"**Sources:**\n{sources}"
            )
            await update.message.reply_text(response_text, parse_mode="Markdown")

        else:
            await update.message.reply_text("❌ Fake news")
    else:
        await update.message.reply_text("❌ Failed to retrieve data from the search engine.")
# ✅ Analyze Image
# ✅ Analyze Image + Caption

# Assuming VISION_API_KEY, GOOGLE_SEARCH_API_KEY, and GOOGLE_CX_ID are already defined


async def analyze_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not update.message.photo:
            await update.message.reply_text("⚠️ Please upload a valid image.")
            return

        caption = update.message.caption.lower() if update.message.caption else ""

        # Get image data
        photo = update.message.photo[-1]
        photo_file = await photo.get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        image_base64 = base64.b64encode(photo_bytes).decode("utf-8")

        # Vision API request
        vision_payload = {
            "requests": [{
                "image": {"content": image_base64},
                "features": [{"type": "WEB_DETECTION"}]
            }]
        }

        url = f"https://vision.googleapis.com/v1/images:annotate?key={VISION_API_KEY}"
        response = requests.post(url, json=vision_payload)
        result = response.json()

        if "error" in result:
            await update.message.reply_text(f"❌ API Error: {result['error']['message']}")
            return

        web_entities = result['responses'][0].get('webDetection', {}).get('webEntities', [])
        detected_terms = [(entity.get('description', '').lower(), entity.get('score', 0)) for entity in web_entities if 'description' in entity]
        entity_keywords = [term for term, _ in detected_terms]

        # Real/Fake logic
        if not detected_terms:
            result_text = "✅ Real Image Detected! (No AI terms found)"
        else:
            confidence_scores = [score for _, score in detected_terms]
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            confidence_variation = max(confidence_scores) - min(confidence_scores)

            if (avg_confidence > HIGH_CONFIDENCE_THRESHOLD and confidence_variation < CONFIDENCE_VARIATION_THRESHOLD) or len(detected_terms) > MIN_TERM_COUNT:
                result_text = (
                    f"❌ Fake or AI-Generated Image Detected!\n"
                    f"➡️ Average Confidence: {avg_confidence:.2f}\n"
                    f"➡️ Confidence Variation: {confidence_variation:.2f}\n"
                    f"➡️ Total Terms Detected: {len(detected_terms)}"
                )
            else:
                result_text = (
                    f"✅ Real Image Detected!\n"
                    f"➡️ Average Confidence: {avg_confidence:.2f}\n"
                    f"➡️ Confidence Variation: {confidence_variation:.2f}\n"
                    f"➡️ Total Terms Detected: {len(detected_terms)}"
                )

            # Enhanced detection
            enhanced_status = "✅ Enhanced/Edited" if confidence_variation > 0.3 and avg_confidence > 0.5 else "❌ Not Enhanced"

        # Caption verification using Google Search API
        caption_result = "ℹ️ No Caption Provided"
        if caption:
            try:
                caption_words = caption.split()
                match_found = False

                async with aiohttp.ClientSession() as session:
                    for word in caption_words:
                        search_url = (
                            f"https://www.googleapis.com/customsearch/v1"
                            f"?q={word}&key={GOOGLE_SEARCH_API_KEY}&cx={GOOGLE_SEARCH_CX}"
                        )
                        async with session.get(search_url) as search_response:
                            data = await search_response.json()
                            if "items" in data:
                                for item in data["items"]:
                                    snippet = item.get("snippet", "").lower()
                                    if any(entity in snippet for entity in entity_keywords):
                                        match_found = True
                                        break
                        if match_found:
                            break

                caption_result = (
                    "✅ Caption Matches Image (Likely True)"
                    if match_found else "❌ Caption Mismatch (Likely False)"
                )
            except Exception as e:
                caption_result = f"❌ Error during caption verification: {str(e)}"

        # Top entity display
        entity_list = "\n".join(
            [f"- {term} (Score: {score:.2f})" for term, score in detected_terms[:10]]
        )

        response_text = (
            f"**🔎 Image Analysis:**\n\n"
            f"{result_text}\n\n"
            f"**Enhanced/Edited:** {enhanced_status}\n"
            f"**Caption Analysis:** {caption_result}\n\n"
            f"**Top Detected Entities:**\n{entity_list or 'No entities detected.'}"
        )

        await update.message.reply_text(response_text, parse_mode="Markdown")

    except Exception as e:
        await update.message.reply_text("❌ An error occurred while processing the image.")
        print(f"[Error] analyze_image: {e}")


# ✅ Start Command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hi! I can detect fake news and AI-generated images.\n\n"
        "📌 *To analyze news:* Send a news statement.\n"
        "📸 *To analyze an image:* Upload an image."
    )

# ✅ Main Function
def main():
    TOKEN = "8295763912:AAG8j92sehUBuU5f9Q5bT_quBLW_q4YGLf4"
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_news))
    app.add_handler(MessageHandler(filters.PHOTO, analyze_image))

    logger.info("🤖 Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
