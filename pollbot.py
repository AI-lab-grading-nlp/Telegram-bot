#!/usr/bin/env python
# pylint: disable=unused-argument, wrong-import-position
# This program is dedicated to the public domain under the CC0 license.

"""
Basic example for a bot that works with polls. Only 3 people are allowed to interact with each
poll/quiz the bot generates. The preview command generates a closed poll/quiz, exactly like the
one the user sends the bot
"""
import logging
from themes import create_clusters
from telegram import __version__ as TG_VER
import os
from chatbot import get_response
from dotenv import load_dotenv
import random

load_dotenv()


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

from themes import themes_pipeline

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

async def poll(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a predefined poll"""
    #questions means themes
    source = context.bot_data.get('source')
    #potential command:
    #df = pd.DataFrame(source, sep=' ', header=None)
    #suggested_themes = create_clusters(df)[1]
    suggested_themes = themes_pipeline(source, 5)
    questions = suggested_themes

    message = await context.bot.send_poll(
        update.effective_chat.id,
        "Which themes would you like to be quizzed on?",
        questions,
        is_anonymous=False,
        allows_multiple_answers=True,
    )
    # Save some info about the poll the bot_data for later use in receive_poll_answer
    payload = {
        message.poll.id: {
            "questions": questions,
            "message_id": message.message_id,
            "chat_id": update.effective_chat.id,
            "answers": 0,
        }
    }
    context.bot_data.update(payload)


async def receive_poll_answer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Summarize a users poll vote"""
    answer = update.poll_answer
    answered_poll = context.bot_data[answer.poll_id]
    try:
        questions = answered_poll["questions"]
    # this means this poll answer update is from an old poll, we can't do our answering then
    except KeyError:
        return
    selected_options = answer.option_ids
    answer_string = ""
    for question_id in selected_options:
        if question_id != selected_options[-1]:
            answer_string += questions[question_id] + " and "
        else:
            answer_string += questions[question_id]
    await context.bot.send_message(
        answered_poll["chat_id"],
        f"{update.effective_user.mention_html()} wants to be quizzed on {answer_string}!",
        parse_mode=ParseMode.HTML,
    )
    context.bot_data['themes'] = answer_string

    answered_poll["answers"] += 1

    context.bot_data['themes'] = answer_string.split(' and ')
    print(context.bot_data['themes'])

async def user_themes(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Save the themes the user wants to be quizzed on in addition to the ones selected"""

    user_themes_string = update.message.text

    # add the themes to the themes list
    context.bot_data['user_themes'] = user_themes_string.split(',')
    context.bot_data['user_themes'] = context.bot_data['user_themes'][1:]
    print(context.bot_data['user_themes'])

    await context.bot.send_message(update.effective_chat.id, "Themes saved!")



async def quiz(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    """Sends a quiz using the source text"""

    source = context.bot_data.get("source")


    prompt = "Bob is a chatbot that gives a multiple choice quiz based on a source text with 4 options labeled from 1 to 4 alongside an explanation for the correct answer in the following format: \n\n Question \n\n 1. option 1 \n 2. option 2 \n 3. option 3 \n 4. option 4  \n\n explanation: ... \n\n The correct answer is labeled with a * \n\n Question \n\n 1. option 1 \n 2. option 2 \n 3. option 3 \n 4. option 4 * \n\n explanation: The correct answer is option 4 because ... . The options are kept less than 100 characters. \n\n The source text is: \n\n"

    prompt += source

    # only add themes if they were included
    if context.bot_data.get("themes"):
        themes = context.bot_data.get("themes")
        if context.bot_data.get("user_themes"):
            themes += context.bot_data.get("user_themes")
        curr_theme = random.sample(themes, k=1)
        print(curr_theme)
        prompt += f'\n\n The themes that Bob should focus on when formulating the questions are: {curr_theme}'

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
    for i in range(4):
        choices[i] = choices[i][:100]

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


async def receive_quiz_answer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Close quiz after three participants took it"""
    # the bot can receive closed poll updates we don't care about
    if update.poll.is_closed:
        return
    if update.poll.total_voter_count == 3:
        try:
            quiz_data = context.bot_data[update.poll.id]
        # this means this poll answer update is from an old poll, we can't stop it then
        except KeyError:
            return
        await context.bot.stop_poll(quiz_data["chat_id"], quiz_data["message_id"])


async def preview(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Ask user to create a poll and display a preview of it"""
    # using this without a type lets the user chooses what he wants (quiz or poll)
    button = [[KeyboardButton("Press me!", request_poll=KeyboardButtonPollType())]]
    message = "Press the button to let the bot generate a preview for your poll"
    # using one_time_keyboard to hide the keyboard
    await update.effective_message.reply_text(
        message, reply_markup=ReplyKeyboardMarkup(button, one_time_keyboard=True)
    )


async def receive_poll(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """On receiving polls, reply to it by a closed poll copying the received poll"""
    actual_poll = update.effective_message.poll
    # Only need to set the question and options, since all other parameters don't matter for
    # a closed poll
    await update.effective_message.reply_poll(
        question=actual_poll.question,
        options=[o.text for o in actual_poll.options],
        # with is_closed true, the poll/quiz is immediately closed
        is_closed=True,
        reply_markup=ReplyKeyboardRemove(),
    )


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Display a help message"""
    await update.message.reply_text("Use /quiz, /poll or /preview to test this bot.")


def main() -> None:
    """Run bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(os.getenv('BOT_TOKEN')).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("source", source))
    application.add_handler(CommandHandler("poll", poll))
    application.add_handler(CommandHandler("quiz", quiz))
    application.add_handler(CommandHandler("user_themes", user_themes))
    application.add_handler(CommandHandler("preview", preview))
    application.add_handler(CommandHandler("help", help_handler))
    application.add_handler(MessageHandler(filters.POLL, receive_poll))
    application.add_handler(PollAnswerHandler(receive_poll_answer))
    application.add_handler(PollAnswerHandler(user_themes))
    application.add_handler(PollHandler(receive_quiz_answer))

    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == "__main__":
    main()