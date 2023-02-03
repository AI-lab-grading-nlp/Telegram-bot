import logging


from telegram import __version__ as TG_VER
import os
from chatbot import get_response

try:

    from telegram import __version_info__

except ImportError:

    __version_info__ = (0, 0, 0, 0, 0)  # type: ignore[assignment]


if __version_info__ < (20, 0, 0, "alpha", 1):

    raise RuntimeError(

        f"This example is not compatible with your current PTB version {TG_VER}. To view the "

        f"{TG_VER} version of this example, "

        f"visit https://docs.python-telegram-bot.org/en/v{TG_VER}/examples.html"

    )

from telegram import (

    KeyboardButton,

    KeyboardButtonPollType,

    Poll,

    ReplyKeyboardMarkup,

    ReplyKeyboardRemove,

    Update,

)

from telegram.constants import ParseMode

from telegram.ext import (

    Application,

    CommandHandler,

    ContextTypes,

    MessageHandler,

    PollAnswerHandler,

    PollHandler,

    filters,

)


# Enable logging

logging.basicConfig(

    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO

)

logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    """Inform user about what this bot can do"""

    await update.message.reply_text(

        "Please give a source text with /source. Use /quiz"

        " to generate a quiz from the source text."

    )


async def source(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    """Save the source text"""

    context.bot_data["source"] = update.message.text

    await update.message.reply_text("Source text saved!")


async def quiz(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    """Send's a quiz using the source text"""

    source = context.bot_data.get("source")

    prompt = "Bob is a chatbot that gives a multiple choice quiz based on a source text with 4 options labeled from 1 to 4 alongside an explanation for the correct answer in the following format: \n\n Question \n\n 1. option 1 \n 2. option 2 \n 3. option 3 \n 4. option 4  \n\n explanation: ... \n\n The correct answer is labeled with a * \n\n Question \n\n 1. option 1 \n 2. option 2 \n 3. option 3 \n 4. option 4 * \n\n explanation: The correct answer is option 4 because ... . The options are kept less than 100 characters. \n\n The source text is: \n\n"

    prompt += source

    prompt += '\n\n Bob: \n\n'

    response = get_response(prompt)

    question = response.index("Question: ")
    choice_1 = response.index("1. ")
    choice_2 = response.index("2. ")
    choice_3 = response.index("3. ")
    choice_4 = response.index("4. ")
    explanation = response.index("Explanation: ")

    question = response[question + 10: choice_1 - 1]
    choice_1 = response[choice_1 + 3: choice_2 - 1]
    choice_2 = response[choice_2 + 3: choice_3 - 1]
    choice_3 = response[choice_3 + 3: choice_4 - 1]
    choice_4 = response[choice_4 + 3: explanation - 1]
    explanation = response[explanation + 13:]

    choices = [choice_1, choice_2, choice_3, choice_4]

    correct_answer = 0

    for i in range(len(choices)):
        if '*' in choices[i]:
            correct_answer = i
            choices[i] = choices[i].replace('*', '')

    # debug_questions = f"Question: {question} \n\n 1. {choice_1} \n 2. {choice_2} \n 3. {choice_3} \n 4. {choice_4} \n\n Explanation: {explanation}"

    # debug_message = await update.effective_chat.send_message(debug_questions)

    message = await update.effective_message.reply_poll(

        question, choices, type=Poll.QUIZ, correct_option_id=correct_answer, is_anonymous=False,
        explanation=explanation
    )

    payload = {

        message.poll.id: {"chat_id": update.effective_chat.id,
                          "message_id": message.message_id}

    }

    context.bot_data.update(payload)


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    """Display a help message"""

    await update.message.reply_text("Use /source to add the piece of text you want to be quized on. Use /quiz to generate a quiz from the source text")


def main() -> None:
    """Run bot."""

    # Create the Application and pass it your bot's token.

    application = Application.builder().token(os.getenv('BOT_TOKEN')).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("source", source))
    application.add_handler(CommandHandler("quiz", quiz))
    application.add_handler(CommandHandler("help", help_handler))

    # Run the bot until the user presses Ctrl-C

    application.run_polling()


if __name__ == "__main__":

    main()
