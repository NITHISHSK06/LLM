"""Prepare the deterministic Exp011 high-quality general-chat SFT dataset."""

from __future__ import annotations

import json
import platform
import random
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tokenizer.tokenizer import SimpleBPETokenizer
from training.instruct_train import InstructionExample, encode_example

SEED = 42
TRAIN_PER_CATEGORY = 500
VAL_PER_CATEGORY = 50
TEST_PER_CATEGORY = 50
CONTEXT_LENGTH = 256
OUTPUT_DIR = PROJECT_ROOT / "data/processed/exp011_general_chat"
TOKENIZER_DIR = PROJECT_ROOT / "tokenizer_exp003"
CATEGORIES = (
    "greetings", "introductions", "wellbeing", "small_talk", "thanks",
    "apologies", "asking_for_help", "offering_help", "positive_emotions",
    "negative_emotions", "encouragement", "agreement", "disagreement",
    "acknowledgement", "farewell", "good_morning", "good_night",
    "casual_questions", "daily_activities", "conversational_followup",
)


SPECS: dict[str, dict[str, tuple[str, ...]]] = {
    "greetings": {
        "train_prompts": ("{d}", "{d}, how are things?", "Hey, {d}.", "Hello there, {d}.", "Nice to see you, {d}."),
        "heldout_prompts": ("Hi! {d}", "Hey there, {d}.", "Good to see you. {d}", "Hello! {d}", "How is it going, {d}?"),
        "train_responses": ("Hi! It is good to hear from you.", "Hey there, I hope your day is going well.", "Hello! Nice to see you too.", "Hi, I am glad you stopped by.", "Hey! How has your day been so far?"),
        "heldout_responses": ("Hello! I am happy to see you.", "Hey, it is nice to hear from you.", "Hi there! I hope things are going smoothly.", "Good to see you. How are you doing?", "Hello! What have you been up to?"),
        "details": ("Hi", "Hello", "Hey", "Hello there", "Hi again", "Hey friend", "Good to see you", "Nice to see you", "Hello for now", "Hey, how is it going"),
    },
    "introductions": {
        "train_prompts": ("I would like to introduce myself: {d}.", "Here is a little about me: {d}.", "I am new here; {d}.", "You may not know me yet: {d}.", "Let me tell you who I am: {d}."),
        "heldout_prompts": ("Just so you know, {d}.", "I wanted to introduce myself: {d}.", "We have not met before; {d}.", "A quick introduction: {d}.", "I am meeting you for the first time; {d}."),
        "train_responses": ("Nice to meet you! I am glad you are here.", "It is lovely to meet you. Thanks for introducing yourself.", "Welcome! I hope you feel comfortable with us.", "Nice to meet you; I would enjoy hearing more about you.", "Hello, it is good to meet you."),
        "heldout_responses": ("It is a pleasure to meet you. Welcome.", "Thanks for telling me; I am glad we have met.", "Nice to meet you! How are you finding things so far?", "Hello, I am pleased to meet you.", "Welcome, and thank you for introducing yourself."),
        "details": ("my name is Riley", "I am joining the team", "I moved here recently", "I am taking this class", "I volunteer with the group", "I work nearby", "I am visiting from another town", "I am helping with the event", "I am your new neighbor", "I am here for the first time"),
    },
    "wellbeing": {
        "train_prompts": ("How are you {d}?", "Are you doing okay {d}?", "How have you been {d}?", "I wanted to check in. How are you {d}?", "You seem quiet. Are you all right {d}?"),
        "heldout_prompts": ("How is everything going {d}?", "Have you been feeling all right {d}?", "How have things been for you {d}?", "You seem thoughtful today; are you okay {d}?", "Can I ask how you are doing {d}?"),
        "train_responses": ("I am doing well, thanks for asking. How about you?", "I have been okay, just taking things one day at a time.", "I am feeling pretty good today.", "I am managing, and I appreciate you checking in.", "Things are going smoothly so far."),
        "heldout_responses": ("I am all right, thank you for asking. How are things with you?", "I have had a full week, but I am doing okay.", "That is kind of you to ask; I am feeling better now.", "I am managing well and trying to keep a steady pace.", "I am okay, and I appreciate having someone check in."),
        "details": ("today", "this morning", "after work", "since yesterday", "after the trip", "before the weekend", "during this busy week", "with everything going on", "after a quiet night", "lately"),
    },
    "small_talk": {
        "train_prompts": ("What do you think about {d}?", "It is nice to talk about {d}, is it not?", "Have you noticed {d}?", "How has {d} been for you?", "I was just thinking about {d}."),
        "heldout_prompts": ("What has {d} been like for you?", "Have you had much experience with {d}?", "I noticed {d}; what do you think?", "Is {d} treating you well?", "What is your take on {d}?"),
        "train_responses": ("It has been pleasant, and it makes the day feel a little easier.", "I noticed that too. It is a nice change of pace.", "That is an easy topic to chat about. What is your view?", "I have enjoyed it more than I expected.", "It has been fairly calm on my side today."),
        "heldout_responses": ("I have enjoyed it, especially when I have time to slow down.", "That is a good observation; I have noticed something similar.", "It has been a pleasant part of the day so far.", "I have not thought about it much, but now I am curious.", "It gives us something easy and interesting to talk about."),
        "details": ("the weather this week", "your weekend plans", "the new cafe", "the morning commute", "the neighborhood", "the music nearby", "the pace of the afternoon", "the view from the window", "the local market", "the quiet evening"),
    },
    "thanks": {
        "train_prompts": ("Thanks for {d}.", "I really appreciate {d}.", "That was kind of you to {d}.", "Thank you so much for {d}.", "I wanted to say thanks for {d}."),
        "heldout_prompts": ("I am grateful that you {d}.", "You have my thanks for {d}.", "I appreciate the fact that you {d}.", "That help meant a lot; thank you for {d}.", "Please know I am thankful that you {d}."),
        "train_responses": ("You are very welcome; I am glad I could help.", "Of course! I was happy to do it.", "No problem at all, and I am glad it helped.", "It was my pleasure. I am happy things worked out.", "Anytime. I appreciate you saying so."),
        "heldout_responses": ("You are welcome. I was glad to be useful.", "It was no trouble, and I am pleased it made a difference.", "My pleasure; I am happy everything worked out.", "Of course. I would be glad to help again.", "Thank you for saying that. It means a lot."),
        "details": ("staying late", "listening to me", "helping with the boxes", "remembering my birthday", "sharing your notes", "checking on the dog", "giving me directions", "making room for me", "bringing dinner", "explaining the problem"),
    },
    "apologies": {
        "train_prompts": ("I am sorry for {d}.", "Please forgive me for {d}.", "I apologize about {d}.", "That was my mistake: {d}.", "I regret {d}, and I am sorry."),
        "heldout_prompts": ("I owe you an apology for {d}.", "I regret {d}; please accept my apology.", "I realize I caused trouble by {d}.", "I am sorry that I ended up {d}.", "I should have handled {d} better."),
        "train_responses": ("I understand. Thank you for apologizing.", "That is all right; I appreciate you taking responsibility.", "I was disappointed, but I am glad we can talk about it.", "Thank you for saying so. Let us work out what happens next.", "I accept your apology, and I hope we can move forward."),
        "heldout_responses": ("I appreciate the apology. Let us find a way to make things right.", "Thank you for being honest; I understand that mistakes happen.", "I accept your apology, and I am glad you addressed it directly.", "It was difficult, but your apology helps.", "Thank you for taking responsibility. We can discuss the next step."),
        "details": ("arriving late", "missing your message", "interrupting you", "forgetting our plan", "speaking too sharply", "leaving the door open", "changing the schedule", "not calling back", "losing the note", "overlooking that detail"),
    },
    "asking_for_help": {
        "train_prompts": ("Could you help me with {d}?", "I need a hand with {d}.", "Would you mind helping me {d}?", "I am not sure how to handle {d}; can you help?", "Do you have time to help me with {d}?"),
        "heldout_prompts": ("Could you give me some guidance about {d}?", "I am stuck with {d}; would you help?", "Would you be able to lend a hand with {d}?", "I could use your advice on {d}.", "Can we look at {d} together?"),
        "train_responses": ("Of course. Tell me what part is giving you trouble.", "I can help with that. Where would you like to start?", "Sure, let us look at it together.", "I would be glad to help; what have you tried so far?", "Absolutely. We can take it one step at a time."),
        "heldout_responses": ("I would be happy to help. Which part feels most difficult?", "Sure, explain what you need and we can work through it.", "Of course; let us start by clarifying the main problem.", "I can take a look. Tell me what you have already tried.", "Yes, we can figure it out together."),
        "details": ("a confusing form", "moving this table", "a computer setting", "a difficult choice", "the new recipe", "the broken shelf", "planning the trip", "organizing these files", "understanding the instructions", "carrying these bags"),
    },
    "offering_help": {
        "train_prompts": ("Would you like help with {d}?", "I can help you {d}, if you want.", "Do you need a hand with {d}?", "Let me know if I can help with {d}.", "I have some time; shall I help with {d}?"),
        "heldout_prompts": ("Could I make {d} easier for you?", "Would it be useful if I helped with {d}?", "I am available if you would like help with {d}.", "Can I take care of part of {d}?", "Would you like me to lend a hand with {d}?"),
        "train_responses": ("That is kind of you. I may take you up on the offer.", "Thanks, a hand with that would be useful.", "I appreciate it. I will let you know if I get stuck.", "Yes, please. It would make things easier.", "That is thoughtful of you; I think I can manage for now."),
        "heldout_responses": ("Thank you for offering. A little help would be welcome.", "That is kind; I will ask if I need an extra hand.", "I appreciate the offer, even if I can finish this myself.", "Yes, please. It would help me get started.", "Thanks for noticing. Let us decide what would be most useful."),
        "details": ("carrying the boxes", "checking the schedule", "setting up the room", "finding the right book", "cleaning up", "calling the office", "sorting the supplies", "learning the controls", "planning dinner", "fixing the loose handle"),
    },
    "positive_emotions": {
        "train_prompts": ("I am happy about {d}.", "I am excited that {d}.", "It made my day when {d}.", "I feel great because of {d}.", "I have good news: {d}."),
        "heldout_prompts": ("I cannot stop smiling about {d}.", "Something wonderful happened: {d}.", "I am feeling especially cheerful because {d}.", "I wanted to share some happy news about {d}.", "I am delighted that {d}."),
        "train_responses": ("That is wonderful to hear! What made the moment special?", "I am happy for you; you deserve to enjoy that feeling.", "That sounds like a lovely reason to smile.", "It is great that something went so well.", "You sound really pleased, and I can see why."),
        "heldout_responses": ("That is lovely news. I can hear how much it means to you.", "I am glad something brought you that much happiness.", "What a nice moment to have; I hope you enjoy it.", "That sounds genuinely exciting. You have every reason to celebrate.", "I am happy for you. What was the best part?"),
        "details": ("my project went well", "the weather cleared up", "a friend called", "I finished the book", "we found a good solution", "the visit went smoothly", "I received a kind message", "the concert was enjoyable", "I reached my goal", "the surprise worked"),
    },
    "negative_emotions": {
        "train_prompts": ("I had a difficult day with {d}.", "I feel upset about {d}.", "I am frustrated by {d}.", "It has been hard dealing with {d}.", "I am disappointed that {d}."),
        "heldout_prompts": ("Today has been rough because of {d}.", "I am having trouble coping with {d}.", "I feel discouraged after {d}.", "Things did not go well with {d}.", "I am struggling a little with {d}."),
        "train_responses": ("I am sorry to hear that. I hope things feel easier soon.", "That sounds difficult; I am here if you want to talk.", "I can understand why you feel worn out.", "I am sorry the day has been so hard. Be kind to yourself.", "That is a lot to carry, and you do not have to handle it alone."),
        "heldout_responses": ("I am sorry it has been so hard. Would talking about it help?", "That sounds exhausting; I hope you can get some rest and support.", "It makes sense that you feel discouraged after that.", "I am here to listen, without trying to dismiss what happened.", "I am sorry you are facing this. What kind of support would help?"),
        "details": ("a tense meeting", "a missed opportunity", "a broken promise", "a long delay", "an argument", "too much work", "a disappointing result", "a confusing message", "a lonely evening", "a plan falling through"),
    },
    "encouragement": {
        "train_prompts": ("I am nervous about {d}.", "I am not sure I can manage {d}.", "I want to give up on {d}.", "I keep struggling with {d}.", "Do you think I can handle {d}?"),
        "heldout_prompts": ("I am losing confidence about {d}.", "I do not know whether I can get through {d}.", "This challenge is making me doubt myself: {d}.", "I could use some encouragement before {d}.", "I am finding {d} harder than expected."),
        "train_responses": ("You have worked hard, and I believe you can handle the next step.", "Keep going; progress does not have to be perfect to matter.", "You are learning as you go, and that is something to be proud of.", "Take it one step at a time. I am rooting for you.", "You have more ability than this difficult moment may suggest."),
        "heldout_responses": ("It is understandable to feel unsure, but your effort already counts for a lot.", "Focus on the next manageable step rather than solving everything at once.", "You do not need to be perfect to make real progress.", "I believe you can learn from this and keep moving forward.", "Give yourself patience; difficult work often takes several attempts."),
        "details": ("the upcoming exam", "starting a new role", "applying for work", "finishing the project", "speaking to the group", "learning the skill", "repairing the bike", "making the decision", "returning to practice", "meeting the deadline"),
    },
    "agreement": {
        "train_prompts": ("I think {d}.", "It seems to me that {d}.", "Would you agree that {d}?", "I feel the same way about {d}.", "My view is that {d}."),
        "heldout_prompts": ("I tend to agree that {d}.", "Does it also seem to you that {d}?", "I have reached the same conclusion: {d}.", "We appear to see this similarly; {d}.", "I think your point about {d} is right."),
        "train_responses": ("I agree. That seems like a sensible way to approach it.", "I feel the same way; your reasoning makes sense to me.", "Yes, I think you are right about that.", "That matches my view as well.", "I agree, especially given the circumstances."),
        "heldout_responses": ("I see it that way too, especially considering the situation.", "Yes, that is close to my own understanding of it.", "I agree; you have made a clear and reasonable point.", "That fits with how I have been thinking about it.", "I share that view, and the reasons seem convincing."),
        "details": ("a quiet evening would help", "the earlier plan is better", "the book is worth reading", "we should ask first", "the change is useful", "this is a fair approach", "the timing makes sense", "a break would help", "the simpler option is best", "we need more time"),
    },
    "disagreement": {
        "train_prompts": ("I am not sure that {d}.", "I see {d} differently.", "Would it be fair to question whether {d}?", "I do not completely agree that {d}.", "There may be another view of {d}."),
        "heldout_prompts": ("I have a different take on {d}.", "I am not convinced that {d}.", "Could we reconsider the idea that {d}?", "My conclusion about {d} is different.", "I understand the point, but {d} may not be so simple."),
        "train_responses": ("I see it a little differently. Could we consider another possibility?", "I am not sure I agree; there may be another side to it.", "That is one view, but I would weigh a few more factors first.", "I understand your point, though my conclusion is different.", "I respectfully disagree because the situation may be more complicated."),
        "heldout_responses": ("I understand the argument, but I would look at the situation another way.", "That is possible, though I think we should consider more information first.", "I see why you say that; my concern is that the details may change the answer.", "I respect that view, but I do not reach the same conclusion.", "Perhaps, although I would prefer to compare the alternatives before deciding."),
        "details": ("the change is unnecessary", "the plan will be easy", "waiting is the only answer", "the first choice is best", "the weather will stay clear", "everyone will agree", "the deadline is flexible", "the cost is unimportant", "we have enough information", "the problem will solve itself"),
    },
    "acknowledgement": {
        "train_prompts": ("The meeting is {d}.", "I wanted you to know that {d}.", "Please note that {d}.", "Here is an update: {d}.", "Just letting you know, {d}."),
        "heldout_prompts": ("A quick update for you: {d}.", "I should mention that {d}.", "You may want to know that {d}.", "The latest information is that {d}.", "For your awareness, {d}."),
        "train_responses": ("Got it, thanks for letting me know.", "Understood. I will keep that in mind.", "Thanks, I have noted the change.", "I see; that is helpful to know.", "All right, I understand what you mean."),
        "heldout_responses": ("Thanks for the update; I understand.", "All right, I have taken note of that.", "That is clear, and I will remember it.", "I appreciate you letting me know.", "Understood. Please tell me if anything else changes."),
        "details": ("the room is ready", "the appointment moved", "the message arrived", "the instructions changed", "the package was delivered", "the schedule is confirmed", "the office is closed", "the plan starts tomorrow", "the key is available", "the form was submitted"),
    },
    "farewell": {
        "train_prompts": ("I should go now; {d}.", "It was nice talking with you; {d}.", "I will see you {d}.", "Thanks for the conversation; I am heading out {d}.", "I need to leave {d}."),
        "heldout_prompts": ("I am going to head off now.", "Let us continue this another time.", "I have to say goodbye {d}.", "I enjoyed our chat, but I should leave.", "I will be on my way {d}."),
        "train_responses": ("Goodbye for now. Take care, and enjoy the rest of your day.", "See you later! It was nice talking with you too.", "Take care; I will see you next time.", "Have a good one, and travel safely.", "Bye for now. Let us catch up again soon."),
        "heldout_responses": ("Take care, and I hope the rest of your day goes well.", "It was good talking with you. See you next time.", "Goodbye for now; I look forward to catching up again.", "Have a peaceful rest of the day.", "See you later, and thank you for the chat."),
        "details": ("for now", "until tomorrow", "before the weekend", "after lunch", "before the next meeting", "at the end of the day", "when the visit ends", "after this conversation", "before dinner", "for the evening"),
    },
    "good_morning": {
        "train_prompts": ("Good morning, {d}!", "Morning! How are you {d}?", "I hope you have a good morning {d}.", "It is a beautiful morning {d}, is it not?", "How is your morning starting {d}?"),
        "heldout_prompts": ("Morning! I hope things are going well {d}.", "Wishing you a pleasant morning {d}.", "How has the day begun for you {d}?", "A bright morning to you {d}.", "I hope you slept well {d}."),
        "train_responses": ("Good morning! I hope your day gets off to a smooth start.", "Morning! I hope you slept well.", "Good morning. It is nice to see you today.", "Morning! Wishing you a calm and productive day.", "Good morning! How is your day beginning?"),
        "heldout_responses": ("Good morning! I hope today feels manageable and pleasant.", "Morning! I am wishing you an easy start to the day.", "Good morning. I hope you have had a restful night.", "Morning, and thank you for the kind greeting.", "Good morning! What are you looking forward to today?"),
        "details": ("today", "this morning", "before breakfast", "when you arrive", "on this weekday", "at the start of the trip", "after a restful night", "before the meeting", "on a quiet morning", "as the day begins"),
    },
    "good_night": {
        "train_prompts": ("Good night, {d}.", "I am heading to bed {d}.", "It has been a long day {d}; I should sleep.", "Sleep well {d}.", "I will see you tomorrow {d}."),
        "heldout_prompts": ("I am winding down for the night {d}.", "Time for me to rest {d}.", "Let us call it a night {d}.", "I hope you have a peaceful night {d}.", "I am off to sleep now {d}."),
        "train_responses": ("Good night. I hope you sleep well and wake up rested.", "Sleep well, and take care until tomorrow.", "Good night! I hope the rest of your evening is peaceful.", "Rest well; you have earned a quiet night.", "Good night. I will see you in the morning."),
        "heldout_responses": ("Good night, and I hope you get the rest you need.", "Sleep peacefully; tomorrow is another chance to begin well.", "Rest well, and thank you for the conversation.", "Good night. I hope your evening ends gently.", "Have a restful night, and see you soon."),
        "details": ("after a long day", "before turning out the light", "at the end of the visit", "when everyone is winding down", "before an early start", "after our chat", "once the work is done", "at the end of the week", "after dinner", "before tomorrow"),
    },
    "casual_questions": {
        "train_prompts": ("What do you like about {d}?", "Have you ever tried {d}?", "What is your opinion of {d}?", "Do you usually enjoy {d}?", "Can you tell me about {d}?"),
        "heldout_prompts": ("How do you feel about {d}?", "What comes to mind when you think of {d}?", "Is {d} something you often choose?", "What has your experience with {d} been?", "Would you recommend {d} to a friend?"),
        "train_responses": ("I enjoy a few different things, depending on the day.", "That depends on the situation, but I am open to trying it.", "I have a favorite, though I am always curious about new options.", "I would be happy to tell you about it. What do you enjoy?", "There are several possibilities, and my answer changes with the season."),
        "heldout_responses": ("I have mixed preferences, but I usually choose what feels relaxing.", "It is a nice topic to discuss; my answer depends on the circumstances.", "I enjoy it occasionally, especially when I have time to spare.", "I can share what I have noticed, though everyone has different tastes.", "I would consider it, particularly if the timing were right."),
        "details": ("a relaxing weekend", "a movie you saw", "a meal you enjoy", "a place to visit", "a hobby to try", "a book to read", "a quiet activity", "a new restaurant", "a favorite season", "a way to spend free time"),
    },
    "daily_activities": {
        "train_prompts": ("What are you doing with {d}?", "How is {d} going?", "Tell me about {d}.", "Do you enjoy {d}?", "What is your usual approach to {d}?"),
        "heldout_prompts": ("How does {d} fit into your day?", "What has {d} been like lately?", "Do you have a routine for {d}?", "When do you usually make time for {d}?", "What do you find useful about {d}?"),
        "train_responses": ("I usually keep it simple and leave room for the day to change.", "It depends on my schedule, but I try to balance useful and enjoyable things.", "That is part of my usual routine, though I sometimes change it.", "I have been trying to make that habit more consistent.", "Most days I manage it by planning a little ahead."),
        "heldout_responses": ("I fit it in when I can and try not to make the routine too rigid.", "It has been going fairly well, although some days are busier than others.", "I usually plan a small amount of time for it and adjust as needed.", "I enjoy it most when I can give it my full attention.", "A little planning makes the activity easier to keep up with."),
        "details": ("studying in the evening", "working from home", "preparing lunch", "getting exercise", "shopping for groceries", "traveling on weekends", "practicing music", "walking after dinner", "organizing the room", "planning the week"),
    },
    "conversational_followup": {
        "train_prompts": ("I just told you about {d}.", "What would you ask after hearing about {d}?", "Let us continue talking about {d}.", "I would like to say more about {d}.", "Did you have a question about {d}?"),
        "heldout_prompts": ("Which part of {d} would you ask about next?", "I mentioned {d}; what would you like to know?", "How could we keep a conversation going about {d}?", "What follow-up would fit after discussing {d}?", "Can you respond naturally to someone describing {d}?"),
        "train_responses": ("That sounds interesting. What happened next?", "I would like to hear more; which part stood out to you?", "How did that make you feel?", "What led you to that decision?", "That gives me a better picture. What happened after that?"),
        "heldout_responses": ("That sounds interesting; what part would you like to explore further?", "I would ask about one detail they mentioned and let them continue.", "A thoughtful question about the experience would keep the conversation moving.", "I would show that I was listening and ask what happened next.", "I would invite them to share whichever part mattered most to them."),
        "details": ("a recent trip", "a new project", "an unusual weekend", "a difficult decision", "an interesting book", "a change of plans", "a family celebration", "a new hobby", "a surprising conversation", "a goal they reached"),
    },
}


def normalize(text: str) -> str:
    replacements = {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"}
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text.casefold()).strip()


def make_detail_variants(details: tuple[str, ...], count: int, profile: str = "train") -> list[str]:
    modifiers = (
        "when I have a quiet moment", "during a busy week", "after making time for it", "when plans are flexible",
        "on an ordinary day", "when I need a change of pace", "after thinking it over", "during a relaxed afternoon",
        "when the timing works", "as part of my routine",
    )
    if profile == "val":
        modifiers = (
            "when the day has been full", "during a slower afternoon", "after a little preparation", "when I get the chance",
            "as things settle down", "when I want to try something different", "after considering the options", "during a free hour",
            "when the moment feels right", "as my schedule allows",
        )
    elif profile == "test":
        modifiers = (
            "after a demanding morning", "while taking a short break", "when there is room to breathe", "as the week unfolds",
            "when I feel ready", "after a change of plans", "during an unhurried hour", "when the opportunity appears",
            "as circumstances permit", "before the day gets busy",
        )
    return [
        f"{details[index % len(details)]} {modifiers[(index // len(details)) % len(modifiers)]}"
        for index in range(count)
    ]


def make_split(split: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    per_category = {"train": TRAIN_PER_CATEGORY, "val": VAL_PER_CATEGORY, "test": TEST_PER_CATEGORY}[split]
    for category in CATEGORIES:
        spec = SPECS[category]
        heldout = split != "train"
        prompt_key = "heldout_prompts" if heldout else "train_prompts"
        response_key = "heldout_responses" if heldout else "train_responses"
        detail_count = per_category // len(spec[prompt_key])
        details = make_detail_variants(spec["details"], detail_count, split)
        for index in range(per_category):
            prompt_template = spec[prompt_key][index % len(spec[prompt_key])]
            detail = details[index // len(spec[prompt_key])]
            prompt = prompt_template.format(d=detail)
            if "{d}" not in prompt_template:
                prompt = f"{prompt} ({detail})"
            response = spec[response_key][index % len(spec[response_key])]
            response = f"{response} {response_closer(category, index)}"
            records.append({"category": category, "prompt": prompt, "response": response})
    return records


def response_closer(category: str, index: int) -> str:
    closers = {
        "greetings": ("I hope your day is going well.", "It is nice to connect.", "How has your day been?", "I am glad we crossed paths.", "I hope things are going smoothly.", "It is good to hear from you.", "What have you been up to?", "I hope you are doing well.", "It is nice to have a moment to talk.", "Let me know how your day goes."),
        "introductions": ("I hope you feel welcome.", "I would enjoy hearing more about you.", "It is good to have you here.", "I hope we get a chance to talk again.", "Thanks for sharing that.", "I look forward to getting to know you.", "It is always nice to meet someone new.", "I hope your first day goes well.", "I am glad we have been introduced.", "Please feel comfortable asking questions."),
        "wellbeing": ("I hope your day continues gently.", "It helps to take things at a steady pace.", "Thanks for asking about me.", "I am taking care of the basics.", "I hope you are doing well too.", "It is good to have someone check in.", "I am trying to keep a balanced pace.", "A little rest usually helps.", "I appreciate your kindness.", "I will let you know if I need support."),
        "small_talk": ("What has your experience been like?", "It is a pleasant thing to notice.", "That gives us something to discuss.", "I would be curious to hear your view.", "It has been nice to think about.", "That makes the day feel lighter.", "I noticed something similar.", "It is an easy topic to enjoy.", "I am glad you brought it up.", "There is always more to notice."),
        "thanks": ("I am glad it helped.", "Your appreciation means a lot.", "I was happy to be useful.", "I would do it again.", "It was no trouble at all.", "I am pleased it made a difference.", "That is kind of you to say.", "I hope everything is easier now.", "I appreciate the thoughtful words.", "It is nice to help when I can."),
        "apologies": ("We can take the next step calmly.", "I appreciate your honesty too.", "It helps to talk about what happened.", "I hope we can move forward.", "Thank you for addressing it directly.", "We can learn from the mistake.", "I value the chance to resolve it.", "Let us focus on what helps now.", "I am glad we discussed it.", "That gives us a better way forward."),
        "asking_for_help": ("We can start with the simplest part.", "A little guidance would make it clearer.", "It helps to work through it together.", "I appreciate your patience.", "One step at a time should be manageable.", "That gives us a good place to begin.", "We can look at the details together.", "I hope the next step feels easier.", "It is useful to talk it through.", "We can adjust if something is unclear."),
        "offering_help": ("I appreciate you thinking of me.", "It is kind of you to notice.", "I will let you know what would help.", "That makes the task feel less daunting.", "I am glad to have the option.", "Your offer is reassuring.", "It is nice to know I am not alone.", "I may ask you in a moment.", "Thanks for being available.", "That gives me some flexibility."),
        "positive_emotions": ("It is worth enjoying the moment.", "I am glad to share the good news.", "That made the effort worthwhile.", "It is nice when things come together.", "I hope the good feeling lasts.", "I wanted you to know about it.", "It gave me a welcome boost.", "That is a moment I will remember.", "I am grateful for the experience.", "It is fun to have something to celebrate."),
        "negative_emotions": ("I am trying to be patient with myself.", "It helps to have someone listen.", "I hope tomorrow feels lighter.", "I am taking the situation one step at a time.", "I appreciate having space to talk.", "Rest may help me regain perspective.", "I do not want to ignore how I feel.", "Small comforts can still matter.", "I am looking for a manageable next step.", "Thank you for hearing me out."),
        "encouragement": ("Small progress still counts.", "You do not have to solve everything today.", "Your effort is already meaningful.", "It is reasonable to take a short break.", "You can learn from each attempt.", "The next step is enough for now.", "You have handled difficult things before.", "It is okay to ask for support.", "Patience can be part of progress.", "I will be cheering you on."),
        "agreement": ("That seems like a reasonable conclusion.", "I am glad we see it similarly.", "It fits the situation well.", "That is a useful way to frame it.", "I can understand that perspective.", "The idea feels practical to me.", "That would be my preference too.", "It is worth keeping in mind.", "I think that approach could work.", "We have a similar view here."),
        "disagreement": ("I would be glad to hear your reasoning.", "There may be useful details to compare.", "I hope we can discuss it calmly.", "It is worth leaving room for another option.", "I understand why the issue is not simple.", "We can look at the evidence together.", "A little more context may help.", "I respect that we see it differently.", "The best choice may depend on the details.", "It is useful to consider both sides."),
        "acknowledgement": ("I will keep that in mind.", "That helps me plan the next step.", "I appreciate the clear update.", "I will let you know if I have questions.", "It is useful to have that information.", "Thanks for keeping me informed.", "I understand what needs to happen.", "That makes the situation clearer.", "I will remember the change.", "Please share any further updates."),
        "farewell": ("I hope the rest of your day goes well.", "Take care until we meet again.", "I enjoyed having the chance to talk.", "I hope the evening is peaceful.", "Travel safely if you are heading out.", "Let us catch up again soon.", "I am glad we had this conversation.", "Have a good rest of the day.", "I hope things go smoothly for you.", "Until next time."),
        "good_morning": ("I hope the day treats you kindly.", "It is a good time for a fresh start.", "I hope you have a calm schedule.", "The morning feels full of possibility.", "I hope something pleasant happens today.", "It is nice to begin the day positively.", "I hope your plans go smoothly.", "A steady start can make a difference.", "I hope you find time to breathe.", "Wishing you a comfortable day ahead."),
        "good_night": ("I hope tomorrow starts gently.", "Rest can make a busy day easier.", "I am glad we had time to talk.", "I hope you wake feeling refreshed.", "The day can end on a peaceful note.", "Take whatever rest you need.", "I hope the night is quiet.", "Tomorrow will bring another opportunity.", "It is good to slow down sometimes.", "Sleep well until we speak again."),
        "casual_questions": ("I would be interested in your answer too.", "It is fun to compare different preferences.", "There is no single right answer.", "That gives us an easy topic to discuss.", "I like hearing how people think about it.", "It depends on the person and the moment.", "I would enjoy exploring that question.", "It is a pleasant thing to wonder about.", "Your answer might change with time.", "That is worth a relaxed conversation."),
        "daily_activities": ("Small routines can make the day easier.", "I try to leave room for changes.", "It helps to keep the plan realistic.", "The details often depend on the day.", "A little preparation usually helps.", "I enjoy finding a rhythm that works.", "It is useful to notice what feels sustainable.", "I adjust when the schedule changes.", "The routine is easier when it stays flexible.", "I am still finding the best balance."),
        "conversational_followup": ("I would be curious to hear more.", "That detail seems worth exploring.", "It sounds like there is more to the story.", "I appreciate the chance to understand it better.", "The next part could change the picture.", "I would listen before offering an opinion.", "That gives us a natural place to continue.", "I like hearing how the situation developed.", "It is helpful to understand the context.", "I would let you decide where to take the story."),
    }
    return closers[category][index % 10]


def validate_records(splits: dict[str, list[dict[str, str]]]) -> dict[str, int]:
    rejected = Counter()
    seen_prompts: dict[str, str] = {}
    seen_pairs: dict[tuple[str, str], str] = {}
    placeholder = re.compile(r"\b(?:lorem|placeholder|todo|sample text|your response)\b|<[^>]+>", re.I)
    for split, records in splits.items():
        for record in records:
            prompt, response, category = record["prompt"], record["response"], record["category"]
            if not isinstance(prompt, str) or not isinstance(response, str) or category not in CATEGORIES:
                rejected["malformed"] += 1
            elif not prompt.strip() or not response.strip():
                rejected["empty"] += 1
            elif len(prompt) < 3 or len(prompt) > 240 or len(response) < 5 or len(response) > 320:
                rejected["length"] += 1
            elif "\n" in prompt or "\n" in response or "User:" in response or "Assistant:" in response:
                rejected["format_or_leakage"] += 1
            elif placeholder.search(prompt + " " + response):
                rejected["placeholder"] += 1
            elif normalize(prompt) == normalize(response):
                rejected["prompt_equals_response"] += 1
            elif re.search(r"\b(\w+)(?:\s+\1){3,}\b", normalize(response)):
                rejected["repeated_words"] += 1
            else:
                prompt_key = normalize(prompt)
                pair_key = (prompt_key, normalize(response))
                if prompt_key in seen_prompts:
                    rejected["duplicate_prompt"] += 1
                elif pair_key in seen_pairs:
                    rejected["duplicate_pair"] += 1
                else:
                    seen_prompts[prompt_key] = split
                    seen_pairs[pair_key] = split
    return dict(rejected)


def token_stats(splits: dict[str, list[dict[str, str]]]) -> dict[str, float | int]:
    tokenizer = SimpleBPETokenizer.load(TOKENIZER_DIR)
    lengths: list[int] = []
    truncations = 0
    for records in splits.values():
        for record in records:
            example = InstructionExample(record["prompt"], record["response"])
            prefix = tokenizer.encode(f"User: {example.prompt}\nAssistant: ", add_bos=True)
            response = tokenizer.encode(example.response, add_eos=True)
            if len(prefix) + len(response) > CONTEXT_LENGTH + 1:
                truncations += 1
            lengths.append(len(encode_example(example, tokenizer, CONTEXT_LENGTH).input_ids))
    return {
        "minimum": min(lengths),
        "maximum": max(lengths),
        "average": mean(lengths),
        "truncated_examples": truncations,
        "truncation_percentage": 100 * truncations / len(lengths),
        "context_length": CONTEXT_LENGTH,
        "vocabulary_size": len(tokenizer.vocab),
    }


def character_stats(records: list[dict[str, str]]) -> dict[str, float | int]:
    prompts = [len(record["prompt"]) for record in records]
    responses = [len(record["response"]) for record in records]
    return {
        "prompt_min": min(prompts), "prompt_max": max(prompts), "prompt_average": mean(prompts),
        "response_min": min(responses), "response_max": max(responses), "response_average": mean(responses),
    }


def overlap_stats(splits: dict[str, list[dict[str, str]]]) -> dict[str, int]:
    prompt_sets = {name: {normalize(record["prompt"]) for record in records} for name, records in splits.items()}
    pair_sets = {name: {(normalize(record["prompt"]), normalize(record["response"])) for record in records} for name, records in splits.items()}
    return {
        "train_val_prompts": len(prompt_sets["train"] & prompt_sets["val"]),
        "train_test_prompts": len(prompt_sets["train"] & prompt_sets["test"]),
        "val_test_prompts": len(prompt_sets["val"] & prompt_sets["test"]),
        "train_val_pairs": len(pair_sets["train"] & pair_sets["val"]),
        "train_test_pairs": len(pair_sets["train"] & pair_sets["test"]),
        "val_test_pairs": len(pair_sets["val"] & pair_sets["test"]),
    }


def report(splits: dict[str, list[dict[str, str]]], rejected: dict[str, int], tokens: dict[str, float | int], overlaps: dict[str, int], response_counts: Counter[str]) -> str:
    lines = [
        "Exp011 High-Quality General Conversation SFT Dataset Report",
        "==============================================================",
        f"Experiment: Exp011 High-Quality General Conversation SFT",
        f"Seed: {SEED}",
        f"Python: {platform.python_version()}",
        f"Total examples: {sum(len(records) for records in splits.values())}",
        f"Train/validation/test: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}",
        f"Rejected example count: {sum(rejected.values())}",
        f"Rejected by reason: {json.dumps(rejected, sort_keys=True)}",
        "",
        "Category distribution:",
    ]
    for split, records in splits.items():
        counts = Counter(record["category"] for record in records)
        lines.append(f"  {split}: {json.dumps(dict(counts), sort_keys=True)}")
        lines.append(f"  {split} character statistics: {json.dumps(character_stats(records), sort_keys=True)}")
    lines.extend([
        "",
        f"Cross-split overlap statistics: {json.dumps(overlaps, sort_keys=True)}",
        f"Tokenized length statistics: {json.dumps(tokens, sort_keys=True)}",
        f"Exact duplicate responses: {sum(count - 1 for count in response_counts.values() if count > 1)}",
        f"Unique normalized responses: {len(response_counts)}",
        "Top 20 normalized responses:",
    ])
    lines.extend(f"  {index}. {response} -> {count}" for index, (response, count) in enumerate(response_counts.most_common(20), 1))
    lines.extend([
        "",
        f"Final dataset paths: {OUTPUT_DIR / 'instruction_train.jsonl'}, {OUTPUT_DIR / 'instruction_val.jsonl'}, {OUTPUT_DIR / 'instruction_test.jsonl'}",
        "No model training was started.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    print("Preparing Exp011 dataset only; model training will not be started.")
    if not TOKENIZER_DIR.exists():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_DIR}")
    rng = random.Random(SEED)
    splits = {name: make_split(name) for name in ("train", "val", "test")}
    for records in splits.values():
        rng.shuffle(records)
    rejected = validate_records(splits)
    if rejected:
        raise ValueError(f"Dataset validation rejected records: {rejected}")
    expected = {"train": 10000, "val": 1000, "test": 1000}
    actual = {name: len(records) for name, records in splits.items()}
    if actual != expected:
        raise ValueError(f"Unexpected counts: {actual}; expected {expected}")
    overlaps = overlap_stats(splits)
    if any(overlaps.values()):
        raise ValueError(f"Cross-split leakage detected: {overlaps}")
    tokens = token_stats(splits)
    if tokens["truncated_examples"]:
        raise ValueError(f"Examples exceed context length: {tokens['truncated_examples']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, records in splits.items():
        path = OUTPUT_DIR / f"instruction_{'val' if split == 'val' else split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        paths[split] = str(path)
    response_counts = Counter(normalize(record["response"]) for records in splits.values() for record in records)
    overlaps = overlap_stats(splits)
    report_text = report(splits, rejected, tokens, overlaps, response_counts)
    (OUTPUT_DIR / "dataset_report.txt").write_text(report_text, encoding="utf-8")
    metadata = {
        "experiment": "Exp011 High-Quality General Conversation SFT",
        "seed": SEED,
        "train_size": 10000,
        "validation_size": 1000,
        "test_size": 1000,
        "categories": list(CATEGORIES),
        "tokenizer": "tokenizer_exp003",
        "context_length": CONTEXT_LENGTH,
        "format": "User/Assistant single-turn",
        "source": "programmatically generated controlled conversational dataset",
        "category_counts": {name: dict(Counter(record["category"] for record in records)) for name, records in splits.items()},
        "overlap_statistics": overlaps,
        "tokenization": tokens,
        "character_statistics": {name: character_stats(records) for name, records in splits.items()},
        "response_diversity": {"unique_normalized_responses": len(response_counts), "top_20": response_counts.most_common(20)},
        "rejected": rejected,
        "paths": paths,
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(report_text)
    for split, records in splits.items():
        print(f"10 deterministic random examples from {split} (seed={SEED}):")
        for record in random.Random(SEED + len(split)).sample(records, 10):
            print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()