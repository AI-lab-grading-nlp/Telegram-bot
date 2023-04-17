#!/usr/bin/env python
# pylint: disable=unused-argument, wrong-import-position
# This program is dedicated to the public domain under the CC0 license.

"""
First, a few callback functions are defined. Then, those functions are passed to
the Application and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.

Usage:
Example of a bot-user conversation using nested ConversationHandlers.
Send /start to initiate the conversation.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""

import logging
from typing import Any, Dict, Tuple

from telegram import __version__ as TG_VER

import os

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
import telegram
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, Poll
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
    PollAnswerHandler,
    PollHandler
)

from themes import themes_pipeline
from chatbot import get_response
from dotenv import load_dotenv
import random

telegram.constants.MAX_OPTION_LENGTH = 600
telegram.constants.MAX_QUESTION_LENGTH = 800
telegram.constants.MAX_EXPLANAION_LENGTH = 600


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Top level conversation states
SELECTING_ACTION, SELECTING_SOURCE, DECIDING_NUMBER_OF_QUESTIONS, SELECTING_THEMES, MAKING_QUIZ, SELECTING_QUIZ = map(
    chr, range(6))

# Source selection conversation states
SAVING_SOURCE = chr(6)

# Number of questions conversation states
SAVING_NUMBER_OF_QUESTIONS = chr(7)
# Theme selection conversation states
AUTOGENERATING_THEMES, SELECTING_THEMES_FROM_LIST, MANUAL_THEME_ENTRY = map(
    chr, range(8, 11))

# Quiz generation conversation states
QUIZ_MADE = chr(12)

# Meta states
STOPPING, SHOWING = map(chr, range(12, 14))
# Shortcut for ConversationHandler.END
END = ConversationHandler.END

# Different constants for this example
(
    START_OVER,
    SOURCE,
    NUMBER_OF_QUESTIONS,
    THEMES,
    QUIZ,
    GOING_TO_MANUAL_THEMES
) = map(chr, range(14, 20))

CURRENT_NUM_QUESTIONS = chr(20)




# Top level conversation callbacks
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    """Conversation hub for choosing between main features of the bot."""
    


    text = (
        "You can add a text source to be quizzed on, select specific topics to focus on, and use those to generate a quiz. \n If you want to stop at any point, type /stop."
    )

    buttons = [
        [
            InlineKeyboardButton(
                text="1. Add a text source", callback_data=str(SELECTING_SOURCE)),
        ],
        [
            InlineKeyboardButton(
                text="2. Decide how many questions you want", callback_data=str(DECIDING_NUMBER_OF_QUESTIONS)),
        ],
        [
            InlineKeyboardButton(
                text="3. Select topics in the source text", callback_data=str(SELECTING_THEMES_FROM_LIST)),
        ],
        [
            InlineKeyboardButton(text="4. Make a quiz",
                                callback_data=str(MAKING_QUIZ)),
            InlineKeyboardButton(text="5. Finish", callback_data=str(END)),
        ]
    ]
    keyboard = InlineKeyboardMarkup(buttons)

    if update.callback_query is None and update.message is None:
        await context.bot.send_message(text=text, reply_markup=keyboard, chat_id=context.user_data['chat_id'])
    elif update.callback_query is None:
        # If we're starting over we don't need to send a new message
        if context.user_data.get(START_OVER):
            await update.message.reply_text(text=text, reply_markup=keyboard)
        else:
            await update.message.reply_text(
                "Hi, I'm Quiz Bot and I'll assist your self-study with autogenerated quizzes."
            )
            await update.message.reply_text(text=text, reply_markup=keyboard)
    else:
        await update.callback_query.answer()
        if context.user_data.get(CURRENT_NUM_QUESTIONS, 0) == context.user_data.get(NUMBER_OF_QUESTIONS, 0):
            await context.bot.send_message(text=text, reply_markup=keyboard, chat_id=context.user_data['chat_id'])
        else:
            await update.callback_query.edit_message_text(
                text=text, reply_markup=keyboard)
    context.user_data[START_OVER] = False
    return SELECTING_ACTION


async def selecting_source(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    '''Give the user the option to either send a file or a text message'''
    text = "Please send the text source you want to be quizzed. \n As of now we support copy-pasted entries. \n If you just want to test out the bot, type 'debug' and we'll send you a random text from our database."

    await update.callback_query.answer()
    await update.callback_query.edit_message_text(text=text)

    return SAVING_SOURCE


async def saving_source(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    '''Save the source to user_data'''

    context.user_data['chat_id'] = update.message.chat_id
    if update.message is not None:
        if update.message.text == 'debug':
            debug_path = os.listdir('test-texts')
            random_file = random.choice(debug_path)
            # opening the file and saving it to source
            source = open(f'test-texts/{random_file}',
                          'r', encoding='utf-8').read()
        else:
            source = update.message.text
    else:
        source = update.message.document.file_id
    context.user_data[SOURCE] = source
    context.user_data[START_OVER] = True
    await update.message.reply_text(f"Source saved")
    return await start(update, context)

import math
import re

async def deciding_number_of_questions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    '''Give the user the option to either send a file or a text message'''
    # get the number of words in the source text. words are separated by punctuation and spaces.
    n_words = len(re.findall(r'\w+', context.user_data[SOURCE]))
    rec_number_of_questions = math.floor(math.log(n_words/69)/0.4)
    rec_number_of_questions = min(10, rec_number_of_questions)

    text = f"Please send the number of questions you want to be quizzed on. \n Based on the length of this text, we suggest {rec_number_of_questions} questions."
    context.user_data[CURRENT_NUM_QUESTIONS] = 0

    await update.callback_query.answer()
    await update.callback_query.edit_message_text(text=text)

    return SAVING_NUMBER_OF_QUESTIONS


async def saving_number_of_questions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    '''Save the source to user_data'''

    context.user_data['chat_id'] = update.message.chat_id
    if update.message is not None:
        number_of_questions = update.message.text
    else:
        number_of_questions = update.message.document.file_id
    try:
        context.user_data[NUMBER_OF_QUESTIONS] = int(number_of_questions)
    except:
        # tell the user they messed up
        await update.message.reply_text("Please try again. Make sure to send a number.")
        return await start(update, context)
    context.user_data[START_OVER] = True
    await update.message.reply_text(f"Number of questions saved")

    return await start(update, context)


async def selecting_themes(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    '''Give the user the option to either autogenerate themes or select from a list'''

    text = "Choose the topics you would like to focus on. \n If you want a combination of bot-generated and self-selected topics, you can choose one option and return to this step again later.'"

    buttons = [
        [
            InlineKeyboardButton(
                text="Let the bot generate topics", callback_data=str(AUTOGENERATING_THEMES)),
            InlineKeyboardButton(
                text="Add topics yourself", callback_data=str(GOING_TO_MANUAL_THEMES)),
        ]
    ]
    keyboard = InlineKeyboardMarkup(buttons)

    await update.callback_query.answer()
    await update.callback_query.edit_message_text(text=text, reply_markup=keyboard)

    return SELECTING_THEMES


async def auto_generating_themes(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    '''Auto generate themes from the source'''

    source = context.user_data[SOURCE]
    suggested_themes = themes_pipeline(source)

    message = await context.bot.send_poll(
        update.effective_chat.id,
        "The bot thought these topics were important. Select all that seem relevant to you.",
        suggested_themes,
        is_anonymous=False,
        allows_multiple_answers=True
    )

    # Save some info about the poll the user_data for later use in receive_poll_answer

    payload = {
        message.poll.id: {
            "questions": suggested_themes,
            "message_id": message.message_id,
            "chat_id": message.chat.id,
            "user_data": context.user_data,
            "answers": 0,
        }
    }

    context.user_data.update(payload)

    return SELECTING_THEMES


async def receive_poll_answer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Summarize a users poll vote"""
    answer = update.poll_answer
    answered_poll = context.user_data[answer.poll_id]
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
    context.user_data['themes'] = answer_string

    answered_poll["answers"] += 1

    context.user_data['themes'] = answer_string.split(' and ')

    return await start(update, context)


async def go_to_manual_theme_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    '''Go to the manual theme entry state'''

    text = "Type in the topics you want to focus on, separated by commas. \n Example: 'Cytoplasm, Anaphase, DNA'"

    await update.callback_query.answer()
    await update.callback_query.edit_message_text(text=text)

    return MANUAL_THEME_ENTRY


async def manual_theme_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    '''Manually input themes'''

    user_themes_string = update.message.text

    # add the themes to the themes list
    context.user_data['user_themes'] = user_themes_string.split(
        ',')
    context.user_data[START_OVER] = True
    await context.bot.send_message(update.effective_chat.id, "Topics saved!")

    return await start(update, context)


async def selecting_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    '''Select the quiz to be made'''

    text = "Click the button below to generate a quiz. If you don't get a response from the bot and the clock icon has disappeared, press the button again."

    buttons = [
        [
            InlineKeyboardButton(
                text="Make a quiz", callback_data=str(QUIZ_MADE)),
        ]
    ]
    keyboard = InlineKeyboardMarkup(buttons)

    await update.callback_query.answer()
    await update.callback_query.edit_message_text(text=text, reply_markup=keyboard)

    return SELECTING_QUIZ

from telegram.constants import ChatAction

async def making_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    '''Make the quiz'''

    source = context.user_data.get(SOURCE)

    prompt = "Bob is chatbot that gives a multiple choice quiz based on a source text. \n\n The correct answer is labeled with * after. There is only one correct answer and the other 3 are incorrect. \n An explanation is given for the correct answer.\n The options are kept less than 100 characters.\n\n The format of the quiz is: \n\n Question: \n\n 1. option 1 \n 2. option 2 \n 3. option 3 \n 4. option 4 * \n\n explanation: Option 4 is correct because ... . \n\n The source text is: \n\n"

    prompt += source

    themes = context.user_data.get("themes", [])

    if context.user_data.get("user_themes"):
        themes += context.user_data.get("user_themes")

    if len(themes) > 0:
        prompt += f'\n This question should be about: {themes[(context.user_data[CURRENT_NUM_QUESTIONS]-1)%len(themes)]} \n\n'

    prompt += '\n\n Bob: \n\n'

    try:
        response = get_response(prompt)
        print
    except:
        return await making_quiz(update, context)

    try:
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
        explanation = response[explanation + 13: explanation + 13 + 200]

        choices = [choice_1, choice_2, choice_3, choice_4]
        for i in range(4):
            choices[i] = choices[i][:100]
    except:
        return await making_quiz(update, context)

    correct_answer = 0

    for i in range(len(choices)):
        if '*' in choices[i]:
            correct_answer = i
            choices[i] = choices[i].replace('*', '')

    try:

        message = await update.effective_message.reply_poll(

            question, choices, type=Poll.QUIZ, correct_option_id=correct_answer, is_anonymous=False,
            explanation=explanation
        )


    except ValueError:
        await update.effective_message.reply_text('Please try again later.')

    payload = {

        message.poll.id: {"chat_id": update.effective_chat.id,
                          "message_id": message.message_id}

    }

    context.user_data[CURRENT_NUM_QUESTIONS] += 1

    context.bot_data.update(payload)

    if context.user_data[CURRENT_NUM_QUESTIONS] < context.user_data[NUMBER_OF_QUESTIONS]:
        return await making_quiz(update, context)

    else:
        return await start(update, context)


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:

    """End Conversation by command."""

    if update.callback_query is not None:

        await update.callback_query.answer()

        text = "See you around!"

        await update.callback_query.edit_message_text(text=text)

        return END

    else:

        await update.message.reply_text("See you around!")

        return END


def main() -> None:
    """Run the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(os.getenv("BOT_TOKEN")).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            SELECTING_ACTION: [
                CallbackQueryHandler(
                    selecting_source, pattern="^" + str(SELECTING_SOURCE) + "$"),
                CallbackQueryHandler(
                    selecting_themes, pattern="^" + str(SELECTING_THEMES_FROM_LIST) + "$"),
                CallbackQueryHandler(
                    selecting_quiz, pattern="^" + str(MAKING_QUIZ) + "$"),
                CallbackQueryHandler(
                    deciding_number_of_questions, pattern="^" +
                    str(DECIDING_NUMBER_OF_QUESTIONS) + "$"
                ),
                CallbackQueryHandler(stop, pattern="^" + str(END) + "$"),
            ],
            SAVING_SOURCE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, saving_source),
            ],
            SAVING_NUMBER_OF_QUESTIONS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND,
                               saving_number_of_questions),
            ],
            SELECTING_THEMES: [
                CallbackQueryHandler(
                    auto_generating_themes, pattern="^" + str(AUTOGENERATING_THEMES) + "$"),
                CallbackQueryHandler(
                    go_to_manual_theme_entry, pattern="^" + str(GOING_TO_MANUAL_THEMES) + "$"),
                PollAnswerHandler(receive_poll_answer)
            ],
            MANUAL_THEME_ENTRY: [
                MessageHandler(filters.TEXT & ~filters.COMMAND,
                               manual_theme_entry)
            ],
            SELECTING_QUIZ: [
                CallbackQueryHandler(
                    making_quiz, pattern="^" + str(QUIZ_MADE) + "$")
            ]
        },
        fallbacks=[CommandHandler("stop", stop)],
        per_chat=False,
    )

    application.add_handler(conv_handler)

    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == "__main__":
    main()
