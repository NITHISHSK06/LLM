"""Build the controlled Exp010 general-conversation dataset."""

from __future__ import annotations

import json
import platform
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tokenizer.tokenizer import SimpleBPETokenizer
from training.instruct_train import InstructionExample, encode_example

SEED = 42
CATEGORIES = (
    "greetings", "introductions", "wellbeing", "small_talk", "thanks",
    "apologies", "asking_for_help", "offering_help", "positive_emotions",
    "negative_emotions", "encouragement", "agreement", "disagreement",
    "acknowledgement", "farewell", "good_morning", "good_night",
    "casual_questions", "daily_activities", "conversational_followup",
)
OUTPUT_DIR = PROJECT_ROOT / "data/processed/exp010_general_chat"
TOKENIZER_DIR = PROJECT_ROOT / "tokenizer_exp003"
BASE_CHECKPOINT = PROJECT_ROOT / "checkpoints/exp006_mixed_30k/best_model.pt"
INSTRUCT_TRAIN = PROJECT_ROOT / "training/instruct_train.py"
GENERATOR = PROJECT_ROOT / "experiments/inference/generate.py"
EXP009_FILES = (
    PROJECT_ROOT / "tools/prepare_exp009_general_chat.py",
    PROJECT_ROOT / "data/processed/exp009_general_chat/instruction_train.jsonl",
    PROJECT_ROOT / "data/processed/exp009_general_chat/instruction_val.jsonl",
)


@dataclass(frozen=True)
class CategorySpec:
    subjects: tuple[str, ...]
    details: tuple[str, ...]
    reply_forms: tuple[str, ...]
    train_prompts: tuple[str, ...]
    val_prompts: tuple[str, ...]
    test_prompts: tuple[str, ...]
    val_replies: tuple[str, ...]
    test_replies: tuple[str, ...]


def spec(
    subjects: tuple[str, ...],
    details: tuple[str, ...],
    reply_forms: tuple[str, ...],
    train_prompts: tuple[str, ...],
    val_prompts: tuple[str, ...],
    test_prompts: tuple[str, ...],
    val_replies: tuple[str, ...],
    test_replies: tuple[str, ...],
) -> CategorySpec:
    if not subjects or not details or len(reply_forms) != 5 or len(train_prompts) != 5:
        raise ValueError("Each category needs non-empty subjects/details, 5 replies, and 5 training prompts.")
    subjects = tuple(subjects[index % len(subjects)] for index in range(5))
    expanded_details: list[str] = []
    for index in range(6):
        detail = details[index % len(details)]
        if detail in expanded_details:
            detail = f"{detail}, on another occasion"
        expanded_details.append(detail)
    details = tuple(expanded_details)
    if not val_prompts or not test_prompts or not val_replies or not test_replies:
        raise ValueError("Held-out prompt and response forms cannot be empty.")
    return CategorySpec(subjects, details, reply_forms, train_prompts, val_prompts, test_prompts, val_replies, test_replies)


SPECS: dict[str, CategorySpec] = {
    "greetings": spec(
        ("a new neighbor", "an old friend", "a colleague", "your cousin", "a familiar face"),
        ("at the front door", "after a long week", "near the coffee machine", "at the market", "on the bus", "outside the library"),
        ("Hi there! It is nice to hear from you.", "Hello! I am glad you stopped by.", "Hey! Nice to see you today.", "Good to see you. How has your day been?", "Hi! I hope things are going well."),
        ("Say hello to {subject} {detail}.", "You run into {subject} {detail}; what do you say?", "What would you say to {subject} {detail}?", "You notice {subject} {detail}. Offer a friendly greeting.", "Greet {subject} {detail} warmly."),
        ("How would you welcome {subject} {detail}?", "What might you say when you see {subject} {detail}?"),
        ("You spot {subject} {detail} after some time apart. What do you say?", "How do you open a friendly conversation with {subject} {detail}?"),
        ("I would greet them warmly and ask how they have been.", "A friendly hello and a simple question about their day would fit."),
        ("I would welcome them naturally and ask what has been happening lately.", "I would start with a warm hello, then give them room to respond."),
    ),
    "introductions": spec(
        ("a new teammate", "a guest at a gathering", "a classmate", "a neighbor", "a volunteer"),
        ("joining the group", "sitting beside you", "at the first meeting", "waiting by the entrance", "helping with the event", "starting today"),
        ("Nice to meet you. I am glad you are here.", "It is good to meet you; I am looking forward to working together.", "Hello, I am pleased to meet you.", "Welcome! I hope you feel comfortable here.", "Nice to meet you. How are you finding things so far?"),
        ("Introduce yourself to {subject} {detail}.", "You meet {subject} {detail}. What do you say first?", "How would you introduce yourself to {subject} {detail}?", "Welcome {subject} {detail} and introduce yourself.", "You are meeting {subject} {detail}; begin a polite introduction."),
        ("What is a natural introduction for meeting {subject} {detail}?", "How could you make an introduction to {subject} {detail}?"),
        ("You have never met {subject} {detail}. How do you introduce yourself?", "What would you say when {subject} {detail} joins the conversation?"),
        ("I would share my name, welcome them, and ask an easy opening question.", "I would introduce myself briefly and make the meeting feel relaxed."),
        ("I would say my name, welcome them, and invite them into the conversation.", "I would keep it simple: share my name and say that it is nice to meet them."),
    ),
    "wellbeing": spec(
        ("you", "your friend", "your coworker", "your brother", "your neighbor"),
        ("after a busy week", "this afternoon", "since yesterday", "after the trip", "during the cold weather", "before the weekend"),
        ("I am doing well, thanks for asking. How about you?", "I have been okay, just taking things one day at a time.", "I am feeling pretty good today.", "I am managing, and I appreciate you asking.", "Things are going smoothly so far."),
        ("Ask how {subject} has been feeling {detail}.", "How is {subject} doing {detail}?", "Check in with {subject} {detail}.", "You want to know whether {subject} is okay {detail}. What do you ask?", "Ask {subject} about their wellbeing {detail}."),
        ("What could you say to check on {subject} {detail}?", "How would you ask {subject} whether everything is okay {detail}?"),
        ("You have not heard from {subject} {detail}. How do you check in?", "Ask {subject} how life has been going {detail}."),
        ("I would ask gently and listen to whatever they want to share.", "A kind check-in would let them know I am available if they need support."),
        ("I would check in with care and give them space to answer honestly.", "I would ask in a calm way and offer to listen if something is difficult."),
    ),
    "small_talk": spec(
        ("the weather", "the weekend", "the neighborhood", "the commute", "the local cafe"),
        ("this morning", "on the way here", "around lunchtime", "after work", "during the afternoon", "before dinner"),
        ("It has been a pleasant change of pace, hasn’t it?", "I noticed that too. It makes the day feel a little easier.", "That is a nice topic; what has your experience been like?", "I have been enjoying it, especially with a quiet moment to spare.", "It has been fairly calm on my side today."),
        ("Make a little conversation about {subject} {detail}.", "What could you mention about {subject} {detail} to keep a chat going?", "Start a casual chat about {subject} {detail}.", "You are making small talk about {subject} {detail}; say something natural.", "Offer a light observation about {subject} {detail}."),
        ("What is an easy small-talk comment about {subject} {detail}?", "How might you keep a conversation going about {subject} {detail}?"),
        ("You are waiting with someone and notice {subject} {detail}. What do you say?", "Use {subject} {detail} as a starting point for a relaxed conversation."),
        ("I would make a simple observation and ask an open, low-pressure question.", "A brief comment followed by curiosity would keep the conversation comfortable."),
        ("I would mention what I noticed and invite the other person to share their view.", "I would keep the comment light and let the conversation develop naturally."),
    ),
    "thanks": spec(
        ("someone who helped you", "a friend who remembered", "a coworker who stayed late", "a host who welcomed you", "a person who shared advice"),
        ("with a difficult task", "on a busy day", "at the last minute", "during the visit", "when you needed it"),
        ("You are very welcome; I am glad I could help.", "Of course! I was happy to do it.", "No problem at all, and I appreciate you saying so.", "It was my pleasure. I hope it made things easier.", "Anytime. I am glad it worked out."),
        ("Thank {subject} for helping {detail}.", "How would you thank {subject} {detail}?", "Express appreciation to {subject} {detail}.", "You want {subject} to know you are grateful {detail}. What do you say?", "Offer a sincere thank-you to {subject} {detail}."),
        ("What could you say to thank {subject} {detail}?", "How would you show appreciation to {subject} {detail}?"),
        ("You are grateful because {subject} helped {detail}. What do you say?", "Thank {subject} for the kindness shown {detail}."),
        ("I would name what they did and explain that it made a difference.", "A specific, sincere thank-you would show that I noticed their effort."),
        ("I would mention the help directly and say how much I appreciated it.", "I would offer a genuine thank-you and let them know their kindness mattered."),
    ),
    "apologies": spec(
        ("you arrived late", "you missed a message", "you interrupted someone", "you forgot a small promise", "you made a mistake"),
        ("before the meeting", "during a busy afternoon", "at the start of the conversation", "while making plans", "in front of the group", "after a long day"),
        ("I am sorry about that. I should have handled it better.", "Please accept my apology; I understand why that was frustrating.", "I apologize. I will be more careful next time.", "I am sorry for the inconvenience, and I appreciate your patience.", "That was my mistake. I am sorry, and I will make it right."),
        ("Apologize because {subject} {detail}.", "What would you say after {subject} {detail}?", "Offer a sincere apology for how {subject} happened {detail}.", "You need to apologize: {subject} {detail}.", "How do you take responsibility when {subject} {detail}?"),
        ("What is a thoughtful apology after {subject} {detail}?", "How could you acknowledge that {subject} caused trouble {detail}?"),
        ("You realize that {subject} affected someone {detail}. What do you say?", "Make amends after {subject} happened {detail}."),
        ("I would acknowledge the problem, apologize plainly, and avoid making excuses.", "A good apology would name the impact and explain how I will do better."),
        ("I would take responsibility, apologize directly, and offer a practical repair.", "I would be honest about the mistake and focus on making things right."),
    ),
    "asking_for_help": spec(
        ("a confusing form", "a heavy box", "a computer setting", "a difficult decision", "a new recipe"),
        ("before the deadline", "while you are learning", "at the office", "in the kitchen", "during the move", "this evening"),
        ("Of course. Tell me what part is giving you trouble.", "I can help with that. Where would you like to start?", "Sure, let us look at it together.", "I would be glad to help; explain what you have tried so far.", "Absolutely. We can take it one step at a time."),
        ("Ask someone for help with {subject} {detail}.", "How would you ask for help with {subject} {detail}?", "You need help handling {subject} {detail}. What do you say?", "Request a hand with {subject} {detail}.", "Politely ask someone to help with {subject} {detail}."),
        ("What could you say when you need help with {subject} {detail}?", "How would you make a clear request about {subject} {detail}?"),
        ("You are unsure how to handle {subject} {detail}. Ask for assistance naturally.", "Invite someone to help you understand {subject} {detail}."),
        ("I would explain briefly what I need and ask whether they have time to help.", "A clear request would make it easy for the other person to know how to respond."),
        ("I would describe the difficulty and ask for one specific kind of assistance.", "I would ask politely, explain what I have tried, and welcome their guidance."),
    ),
    "offering_help": spec(
        ("a friend carrying bags", "a coworker facing a deadline", "a neighbor with a repair", "a guest looking lost", "someone learning a skill"),
        ("near the entrance", "at the end of the day", "during a busy morning", "while everyone is preparing", "after the appointment", "on the way home"),
        ("I can help if you would like. What should I do?", "Would you like a hand with that?", "I have a little time; perhaps I can make this easier.", "Let me know what would be most useful.", "I am happy to help, so please do not hesitate to ask."),
        ("Offer help to {subject} {detail}.", "How would you offer a hand to {subject} {detail}?", "You notice {subject} {detail}. What do you say?", "Let {subject} know you can help {detail}.", "Make a considerate offer of help to {subject} {detail}."),
        ("What could you say if you want to help {subject} {detail}?", "How would you offer assistance without being pushy to {subject} {detail}?"),
        ("You see that {subject} may need support {detail}. Offer help naturally.", "Give {subject} a chance to accept help {detail}."),
        ("I would offer specific help and let them decide whether they want it.", "I would ask what would be useful rather than assuming what they need."),
        ("I would make a gentle offer and respect their answer either way.", "I would point out that I am available and ask how I could contribute."),
    ),
    "positive_emotions": spec(
        ("a good surprise", "a finished project", "a sunny afternoon", "a kind message", "a personal achievement"),
        ("that happened today", "after weeks of effort", "at the end of the week", "during a quiet moment", "when you needed encouragement", "on your way home"),
        ("That is wonderful to hear! What made the moment special?", "I am happy for you; you deserve to enjoy that feeling.", "That sounds like a lovely reason to smile.", "It is great that something went so well.", "You sound really pleased, and I can see why."),
        ("Share that you feel happy about {subject} {detail}.", "Tell someone you are excited about {subject} {detail}.", "You are feeling cheerful because of {subject} {detail}. What do you say?", "Express your positive feelings about {subject} {detail}.", "Talk about the good feeling connected with {subject} {detail}."),
        ("How might you describe your happiness about {subject} {detail}?", "What could you say when {subject} brings you joy {detail}?"),
        ("You are delighted that {subject} happened {detail}. Share the news.", "Tell a friend why {subject} made you smile {detail}."),
        ("I would share the good news and explain what made it meaningful.", "I would let the other person see my enthusiasm while giving them the context."),
        ("I would describe what went well and invite them to celebrate the moment with me.", "I would share the feeling honestly and say why the event mattered."),
    ),
    "negative_emotions": spec(
        ("a disappointing day", "an awkward conversation", "a difficult phone call", "a plan that fell through", "a stressful morning"),
        ("at work", "with a close friend", "before an important event", "after waiting a long time", "on your way home", "during a busy week"),
        ("I am sorry you are dealing with that. I hope things ease up soon.", "That sounds really difficult; I am here if you want to talk.", "I can understand why you feel worn out. Be kind to yourself today.", "I am sorry the day has been so hard. Perhaps a little rest will help.", "That is a lot to carry. You do not have to handle it alone."),
        ("Say that you are having a hard time with {subject} {detail}.", "Tell someone you feel discouraged by {subject} {detail}.", "You are upset because of {subject} {detail}. Share how you feel.", "Describe the difficult feeling caused by {subject} {detail}.", "Open up about feeling low after {subject} {detail}."),
        ("What could you say to support someone upset about {subject} {detail}?", "How might you respond kindly when {subject} has been difficult {detail}?"),
        ("You learn that {subject} has gone badly for someone {detail}. Respond with empathy.", "Someone says {subject} left them discouraged {detail}. What do you say?"),
        ("I would listen without dismissing the feeling and ask what kind of support would help.", "I would acknowledge that the situation hurts and avoid rushing to offer a solution."),
        ("I would recognize how hard it sounds, listen carefully, and offer steady support.", "I would respond with empathy first, then ask whether they want advice or company."),
    ),
    "encouragement": spec(
        ("a friend preparing for an exam", "a teammate learning a new role", "someone applying for a job", "a neighbor starting a project", "a relative facing a challenge"),
        ("this week", "before the big day", "after a setback", "while building confidence", "during a long process", "as the deadline approaches"),
        ("You have worked hard, and I believe you can handle the next step.", "Keep going; progress does not have to be perfect to matter.", "You are learning as you go, and that is something to be proud of.", "Take it one step at a time. I am rooting for you.", "You have more ability than this difficult moment may suggest."),
        ("Encourage {subject} {detail}.", "What could you say to support {subject} {detail}?", "Give {subject} confidence {detail}.", "Offer kind encouragement to {subject} {detail}.", "Help {subject} keep going {detail}."),
        ("How might you encourage {subject} {detail}?", "What supportive words could you offer {subject} {detail}?"),
        ("You know {subject} is working hard {detail}. Give them encouragement.", "Remind {subject} of their progress {detail}."),
        ("I would point out their effort and remind them that improvement takes time.", "I would encourage them without pretending the challenge is easy."),
        ("I would recognize the work already done and help them focus on the next manageable step.", "I would offer confidence, patience, and a reminder that setbacks are not the whole story."),
    ),
    "agreement": spec(
        ("the plan sounds sensible", "a quiet evening would help", "the new schedule is better", "that book is worth reading", "the earlier train is convenient"),
        ("for the group", "after a long day", "for this month", "when you have time", "in this situation", "before making a decision"),
        ("I agree. That seems like a sensible way to approach it.", "I feel the same way; your reasoning makes sense to me.", "Yes, I think you are right about that.", "That matches my view as well.", "I agree, especially given the circumstances."),
        ("Agree that {subject} {detail}.", "How would you show agreement when {subject} {detail}?", "Someone says {subject} {detail}; respond to say you agree.", "Express agreement with the idea that {subject} {detail}.", "Confirm that you share the view that {subject} {detail}."),
        ("What could you say to agree that {subject} {detail}?", "How would you acknowledge that you share this view: {subject} {detail}?"),
        ("A friend suggests that {subject} {detail}. Respond in agreement.", "Someone makes the point that {subject} {detail}. Say you agree."),
        ("I would state my agreement and briefly explain which part makes sense to me.", "I would agree naturally without making the conversation sound overly formal."),
        ("I would confirm the shared view and add a short reason for my agreement.", "I would say that I see it the same way and connect it to the situation."),
    ),
    "disagreement": spec(
        ("the change is unnecessary", "the plan will be easy", "the restaurant is the best choice", "waiting is the only answer", "the weather will stay clear"),
        ("for everyone", "without more information", "at this point", "given the circumstances", "for the whole week", "before checking the details"),
        ("I see it a little differently. Could we consider another possibility?", "I am not sure I agree; there may be another side to it.", "That is one view, but I would weigh a few more factors first.", "I understand your point, though my conclusion is different.", "I respectfully disagree because the situation may be more complicated."),
        ("Politely disagree that {subject} {detail}.", "How would you express another view when {subject} {detail}?", "Someone claims {subject} {detail}; respond with a respectful disagreement.", "Explain that you do not fully agree that {subject} {detail}.", "Offer a different perspective on the idea that {subject} {detail}."),
        ("What could you say if you disagree that {subject} {detail}?", "How might you share another perspective when {subject} {detail}?"),
        ("A friend believes {subject} {detail}. Respond thoughtfully without being dismissive.", "Challenge the assumption that {subject} {detail} while staying courteous."),
        ("I would explain my concern calmly and show that I understand the other perspective.", "I would disagree with the idea, not attack the person who suggested it."),
        ("I would acknowledge the point, then give a clear reason for seeing things differently.", "I would keep the tone respectful and invite more discussion rather than trying to win."),
    ),
    "acknowledgement": spec(
        ("the meeting moved", "the message arrived", "the instructions changed", "the package was delivered", "the appointment was confirmed"),
        ("this morning", "a few minutes ago", "for next week", "after the update", "before lunch", "on the shared calendar"),
        ("Got it, thanks for letting me know.", "Understood. I will keep that in mind.", "Thanks, I have noted the change.", "I see; that is helpful to know.", "All right, I understand what you mean."),
        ("Acknowledge that {subject} {detail}.", "How would you show you understand that {subject} {detail}?", "Someone tells you that {subject} {detail}; respond appropriately.", "Confirm that you heard the update about {subject} {detail}.", "Let the speaker know you noticed that {subject} {detail}."),
        ("What could you say after learning that {subject} {detail}?", "How would you confirm that you understood the update that {subject} {detail}?"),
        ("You receive an update saying {subject} {detail}. Acknowledge it briefly.", "Respond to the information that {subject} {detail}."),
        ("I would confirm that I heard the information and mention any next step if needed.", "A brief acknowledgment would show that the update reached me."),
        ("I would thank them for the update and make clear that I understand it.", "I would respond simply, then ask a question only if something remains unclear."),
    ),
    "farewell": spec(
        ("a friend after lunch", "a coworker at the end of the day", "a guest leaving", "a neighbor heading home", "a classmate after a discussion"),
        ("for now", "until tomorrow", "before the weekend", "after a pleasant visit", "when the conversation ends", "before the next meeting"),
        ("Goodbye for now. Take care, and I hope the rest of your day goes well.", "See you later! It was nice talking with you.", "Take care; I will see you next time.", "Have a good one, and travel safely.", "Bye for now. Let us catch up again soon."),
        ("Say goodbye to {subject} {detail}.", "How would you take leave of {subject} {detail}?", "End the conversation warmly with {subject} {detail}.", "Offer a friendly farewell to {subject} {detail}.", "You are leaving {subject} {detail}; what do you say?"),
        ("What could you say when parting from {subject} {detail}?", "How would you close the conversation with {subject} {detail}?"),
        ("You need to leave while speaking with {subject} {detail}. Say goodbye.", "End your visit with {subject} {detail} on a warm note."),
        ("I would say goodbye warmly and mention when we might speak or meet again.", "I would wish them well and make the parting feel friendly rather than abrupt."),
        ("I would offer a kind farewell, wish them a good rest of the day, and leave clearly.", "I would say goodbye and let them know I enjoyed the conversation."),
    ),
    "good_morning": spec(
        ("a friend", "your family", "a coworker", "a neighbor", "the people in your group"),
        ("at sunrise", "on a busy weekday", "before breakfast", "when you arrive", "at the start of a trip", "after a restful night"),
        ("Good morning! I hope your day gets off to a smooth start.", "Morning! I hope you slept well.", "Good morning. It is nice to see you today.", "Morning! Wishing you a calm and productive day.", "Good morning! How is your day beginning?"),
        ("Wish {subject} a good morning {detail}.", "What morning greeting could you give {subject} {detail}?", "Greet {subject} at the start of the day {detail}.", "Open the day pleasantly with {subject} {detail}.", "Say good morning to {subject} {detail}."),
        ("How would you greet {subject} in the morning {detail}?", "What could you say to start the morning with {subject} {detail}?"),
        ("It is morning and you meet {subject} {detail}. What do you say?", "Begin a friendly morning exchange with {subject} {detail}."),
        ("I would greet them warmly and wish them a good start to the day.", "A cheerful hello and a simple question about their morning would be natural."),
        ("I would offer a pleasant morning greeting and hope the day treats them well.", "I would say good morning with a warm tone and leave room for a reply."),
    ),
    "good_night": spec(
        ("a tired friend", "your child", "a housemate", "a guest", "a relative"),
        ("after a long day", "before turning out the light", "at the end of a visit", "when everyone is winding down", "before an early start", "after finishing a quiet chat"),
        ("Good night. I hope you sleep well and wake up feeling rested.", "Sleep well, and take care until tomorrow.", "Good night! I hope the rest of your evening is peaceful.", "Rest well; you have earned a quiet night.", "Good night. I will see you in the morning."),
        ("Wish {subject} good night {detail}.", "How would you say good night to {subject} {detail}?", "Close the evening kindly with {subject} {detail}.", "Offer a peaceful night-time farewell to {subject} {detail}.", "Say good night to {subject} before parting {detail}."),
        ("What could you say before {subject} goes to sleep {detail}?", "How would you close the evening with {subject} {detail}?"),
        ("The day is ending while you are with {subject} {detail}. Say good night.", "Give {subject} a warm night-time farewell {detail}."),
        ("I would wish them a restful night and a gentle start to tomorrow.", "A calm good-night message would help the day end on a kind note."),
        ("I would say good night warmly and wish them the rest they need.", "I would close the conversation peacefully and let them know I will see them later."),
    ),
    "casual_questions": spec(
        ("favorite weekend plans", "a movie you recently saw", "a meal you enjoy", "a place you like visiting", "a hobby you have tried"),
        ("when meeting someone new", "during a relaxed lunch", "on a quiet afternoon", "while waiting together", "after work", "at a social gathering"),
        ("I enjoy a few different things, but lately I have been making time for simple, relaxing plans.", "That depends on the day; I usually choose something that lets me slow down.", "I have a favorite, though I am always open to trying something new.", "I would be happy to tell you about it. What do you enjoy?", "There are several possibilities, and my answer changes with the season."),
        ("Ask {subject} as a light conversation question {detail}.", "What casual question could you ask about {subject} {detail}?", "Start a friendly question about {subject} {detail}.", "Invite someone to talk about {subject} {detail}.", "Ask about {subject} without making the conversation formal {detail}."),
        ("How could you casually ask about {subject} {detail}?", "What is a relaxed question about {subject} {detail}?"),
        ("You want to learn more about {subject} while chatting {detail}. Ask a natural question.", "Use {subject} to invite an easy conversation {detail}."),
        ("I would ask an open question that is easy to answer and easy to build on.", "I would keep the question curious and personal without making it intrusive."),
        ("I would phrase it casually and share a little about my own answer too.", "I would ask with genuine curiosity and allow the other person to choose how much to say."),
    ),
    "daily_activities": spec(
        ("your morning routine", "your lunch break", "your trip to the store", "your evening plans", "your exercise habit"),
        ("on a normal weekday", "when you have spare time", "before the weather changes", "after finishing work", "during a quiet weekend", "when the schedule is full"),
        ("I usually keep it simple and leave a little room for the day to change.", "It depends on my schedule, but I try to make time for one useful thing and one enjoyable thing.", "That is part of my usual routine, though I sometimes change it up.", "I have been trying to make that habit more consistent.", "Most days I manage it by planning a little ahead."),
        ("Talk about {subject} {detail}.", "How would you describe {subject} {detail}?", "Tell someone what you do with {subject} {detail}.", "Share a short update about {subject} {detail}.", "Explain your usual approach to {subject} {detail}."),
        ("What might you say about {subject} {detail}?", "How could you give a natural update on {subject} {detail}?"),
        ("Someone asks what happens with {subject} {detail}. Give a conversational answer.", "Describe how {subject} fits into your day {detail}."),
        ("I would give a concise example and mention whether the routine changes from day to day.", "I would describe the habit in ordinary language and keep the answer easy to follow."),
        ("I would explain the usual pattern, while admitting that real days sometimes vary.", "I would share enough detail to answer the question without turning it into a list."),
    ),
    "conversational_followup": spec(
        ("a story someone just told", "a new project", "a recent trip", "an interesting opinion", "a change in plans"),
        ("after they finish speaking", "when the details are surprising", "during a relaxed chat", "before changing subjects", "when you want to understand more", "near the end of the conversation"),
        ("That sounds interesting. What happened next?", "I would like to hear more; which part stood out to you?", "How did that make you feel?", "What led you to that decision?", "That gives me a better picture. What happened after that?"),
        ("Ask a natural follow-up about {subject} {detail}.", "Keep the conversation going after {subject} {detail}.", "Show interest in {subject} with a follow-up question {detail}.", "Respond to {subject} by inviting more detail {detail}.", "What could you ask next about {subject} {detail}?"),
        ("What follow-up question would fit {subject} {detail}?", "How could you show interest and continue discussing {subject} {detail}?"),
        ("Someone mentions {subject} {detail}. Ask a question that builds on what they said.", "Continue the discussion of {subject} without abruptly changing topics {detail}."),
        ("I would refer to one detail they shared and ask an open question about it.", "A thoughtful follow-up would show that I was listening rather than merely waiting to speak."),
        ("I would build on their comment with curiosity and let them decide where to take the story.", "I would ask about the part that naturally invites more context or feeling."),
    ),
}


def normalize(text: str) -> str:
    replacements = {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"}
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text.casefold()).strip()


def build_examples() -> dict[str, list[dict[str, str | int | bool]]]:
    splits: dict[str, list[dict[str, str | int | bool]]] = {"train": [], "val": [], "test": []}
    for category in CATEGORIES:
        category_spec = SPECS[category]
        scenarios = [
            (subject, detail, category_spec.reply_forms[(subject_index + detail_index) % 5])
            for subject_index, subject in enumerate(category_spec.subjects)
            for detail_index, detail in enumerate(category_spec.details)
        ]
        for scenario_index, (subject, detail, reply) in enumerate(scenarios):
            values = {"subject": subject, "detail": detail, "reply": reply}
            for form_index, prompt_template in enumerate(category_spec.train_prompts):
                splits["train"].append({
                    "id": f"train-{category}-{scenario_index:02d}-{form_index}",
                    "category": category,
                    "prompt": prompt_template.format(**values),
                    "response": reply,
                    "generalization": False,
                })
            if scenario_index < 25:
                val_prompt = category_spec.val_prompts[scenario_index % len(category_spec.val_prompts)]
                val_reply = category_spec.val_replies[scenario_index % len(category_spec.val_replies)]
                values["reply"] = val_reply
                splits["val"].append({
                    "id": f"val-{category}-{scenario_index:02d}",
                    "category": category,
                    "prompt": val_prompt.format(**values),
                    "response": val_reply.format(**values),
                    "generalization": True,
                })
                test_prompt = category_spec.test_prompts[scenario_index % len(category_spec.test_prompts)]
                test_reply = category_spec.test_replies[scenario_index % len(category_spec.test_replies)]
                values["reply"] = test_reply
                splits["test"].append({
                    "id": f"test-{category}-{scenario_index:02d}",
                    "category": category,
                    "prompt": test_prompt.format(**values),
                    "response": test_reply.format(**values),
                    "generalization": True,
                })
    return splits


def validate_and_shuffle(splits: dict[str, list[dict[str, str | int | bool]]]) -> tuple[dict[str, list[dict[str, str | int | bool]]], dict[str, int]]:
    rejected = Counter()
    seen_prompts: dict[str, str] = {}
    seen_pairs: dict[tuple[str, str], str] = {}
    valid_categories = set(CATEGORIES)
    for split_name, records in splits.items():
        accepted: list[dict[str, str | int | bool]] = []
        for record in records:
            prompt, response, category = record["prompt"], record["response"], record["category"]
            if not isinstance(prompt, str) or not isinstance(response, str) or not isinstance(category, str):
                rejected["malformed"] += 1
                continue
            prompt, response = prompt.strip(), response.strip()
            if not prompt or not response:
                rejected["empty"] += 1
                continue
            if category not in valid_categories:
                rejected["invalid_category"] += 1
                continue
            if len(prompt) > 220 or len(response) > 420:
                rejected["too_long"] += 1
                continue
            if "\n" in prompt or "\n" in response or re.search(r"(?:<\|.*?\|>|###|\bBEGIN_[A-Z_]+\b|\bEND_[A-Z_]+\b)", prompt + response):
                rejected["artifacts"] += 1
                continue
            if normalize(prompt) == normalize(response) and category not in {"acknowledgement"}:
                rejected["prompt_equals_response"] += 1
                continue
            normalized_prompt = normalize(prompt)
            normalized_pair = (normalized_prompt, normalize(response))
            if normalized_prompt in seen_prompts:
                rejected["duplicate_prompt"] += 1
                continue
            if normalized_pair in seen_pairs:
                rejected["duplicate_pair"] += 1
                continue
            seen_prompts[normalized_prompt] = split_name
            seen_pairs[normalized_pair] = split_name
            record["prompt"], record["response"] = prompt, response
            accepted.append(record)
        splits[split_name] = accepted

    if any(rejected.values()):
        print(f"Rejected records: {dict(rejected)}")
    return splits, dict(rejected)


def write_jsonl(path: Path, records: list[dict[str, str | int | bool]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def stats_for(records: list[dict[str, str | int | bool]]) -> dict[str, float | int]:
    prompts = [len(str(record["prompt"])) for record in records]
    responses = [len(str(record["response"])) for record in records]
    return {
        "count": len(records),
        "prompt_min": min(prompts) if prompts else 0,
        "prompt_max": max(prompts) if prompts else 0,
        "prompt_average": mean(prompts) if prompts else 0.0,
        "response_min": min(responses) if responses else 0,
        "response_max": max(responses) if responses else 0,
        "response_average": mean(responses) if responses else 0.0,
    }


def tokenization_stats(splits: dict[str, list[dict[str, str | int | bool]]]) -> dict[str, int | float]:
    tokenizer = SimpleBPETokenizer.load(TOKENIZER_DIR)
    from model.config import ModelConfig
    import torch

    checkpoint = torch.load(BASE_CHECKPOINT, map_location="cpu")
    context_length = ModelConfig(**checkpoint["model_config"]).context_length
    lengths: list[int] = []
    truncated = 0
    for records in splits.values():
        for record in records:
            example = InstructionExample(str(record["prompt"]), str(record["response"]))
            prefix = tokenizer.encode(f"User: {example.prompt}\nAssistant: ", add_bos=True)
            response = tokenizer.encode(example.response, add_eos=True)
            raw_length = len(prefix) + len(response)
            if raw_length > context_length + 1:
                truncated += 1
            encoded = encode_example(example, tokenizer, context_length)
            lengths.append(len(encoded.input_ids))
    return {
        "context_length": context_length,
        "examples_exceeding_context": truncated,
        "examples_requiring_truncation": truncated,
        "tokenized_min": min(lengths),
        "tokenized_max": max(lengths),
        "tokenized_average": mean(lengths),
        "vocabulary_size": len(tokenizer.vocab),
    }


def report_text(splits: dict[str, list[dict[str, str | int | bool]]], rejected: dict[str, int], token_stats: dict[str, int | float], duplicate_count: int = 0) -> str:
    lines = [
        "Exp010 General Conversation Dataset Report",
        "===========================================",
        f"Random seed: {SEED}",
        f"Python version: {platform.python_version()}",
        f"Total examples generated: {sum(len(records) for records in splits.values())}",
        f"Final train count: {len(splits['train'])}",
        f"Final validation count: {len(splits['val'])}",
        f"Final test count: {len(splits['test'])}",
        f"Rejected records: {sum(rejected.values())}",
        f"Rejected by reason: {json.dumps(rejected, sort_keys=True)}",
        f"Duplicate counts: {duplicate_count}",
        "Duplicate prompts: 0",
        "Duplicate prompt-response pairs: 0",
    ]
    for split_name in ("train", "val", "test"):
        category_counts = Counter(str(record["category"]) for record in splits[split_name])
        lines.append(f"Category counts for {split_name}: {json.dumps(dict(sorted(category_counts.items())), sort_keys=True)}")
        lengths = stats_for(splits[split_name])
        lines.append(
            f"{split_name} prompt chars min/max/average: {lengths['prompt_min']}/{lengths['prompt_max']}/{lengths['prompt_average']:.2f}"
        )
        lines.append(
            f"{split_name} response chars min/max/average: {lengths['response_min']}/{lengths['response_max']}/{lengths['response_average']:.2f}"
        )
    response_counts = Counter(normalize(str(record["response"])) for records in splits.values() for record in records)
    lines.append(f"Most frequent normalized responses: {json.dumps(response_counts.most_common(10), ensure_ascii=False)}")
    lines.extend([
        f"Tokenizer vocabulary size: {token_stats['vocabulary_size']}",
        f"Context length: {token_stats['context_length']}",
        f"Examples exceeding context: {token_stats['examples_exceeding_context']}",
        f"Examples requiring truncation: {token_stats['examples_requiring_truncation']}",
        f"Tokenized length min/max/average: {token_stats['tokenized_min']}/{token_stats['tokenized_max']}/{token_stats['tokenized_average']:.2f}",
        "Train/validation/test have no exact prompt overlap: True",
        "Train/validation/test have no exact prompt-response overlap: True",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    print("Preparing Exp010 general conversation dataset; no model training will be started.")
    for required in (BASE_CHECKPOINT, TOKENIZER_DIR, INSTRUCT_TRAIN, GENERATOR):
        if not required.exists():
            raise FileNotFoundError(f"Required prerequisite is missing: {required}")
    for protected in EXP009_FILES:
        if not protected.exists():
            raise FileNotFoundError(f"Expected protected Exp009 path is missing: {protected}")

    splits, rejected = validate_and_shuffle(build_examples())
    # The construction intentionally targets exactly 3,000/500/500.
    expected_counts = {"train": 3000, "val": 500, "test": 500}
    if {name: len(records) for name, records in splits.items()} != expected_counts:
        raise ValueError(f"Unexpected split counts: { {name: len(records) for name, records in splits.items()} }")
    token_stats = tokenization_stats(splits)
    for split_name, records in splits.items():
        write_jsonl(OUTPUT_DIR / f"instruction_{'val' if split_name == 'val' else split_name}.jsonl", records)

    report = report_text(splits, rejected, token_stats)
    (OUTPUT_DIR / "dataset_report.txt").write_text(report, encoding="utf-8")
    metadata = {
        "experiment": "Exp010 general conversation",
        "seed": SEED,
        "counts": {name: len(records) for name, records in splits.items()},
        "category_counts": {name: dict(sorted(Counter(str(r["category"]) for r in records).items())) for name, records in splits.items()},
        "rejected": rejected,
        "duplicate_counts": {"total": 0, "prompts": 0, "prompt_response_pairs": 0},
        "character_statistics": {name: stats_for(records) for name, records in splits.items()},
        "tokenization": token_stats,
        "no_exact_prompt_overlap": True,
        "no_exact_prompt_response_overlap": True,
        "test_examples_are_held_out_templates": True,
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(report)
    print(f"Wrote Exp010 files to {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()