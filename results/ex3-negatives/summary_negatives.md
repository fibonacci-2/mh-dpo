# Ex3 negative-generation QC summary

Total generated rows: 624. Flagged (heuristic, needs review): 128.

Flags are surface-form heuristics (keyword/length checks), not a model judgment -- treat a flag as "worth a human look," not a confirmed failure.

## Pass rate by component

| component | n | flagged | pass rate |
|---|---|---|---|
| cr_distortion_reinforcement | 156 | 50 | 68% |
| cr_toxic_positivity | 156 | 59 | 62% |
| ev_ablation | 156 | 5 | 97% |
| sq_ablation | 156 | 14 | 91% |

## Sample flagged rows

### ex02_2_b_2 / sq_ablation (flags: still_contains_questions)

**Client statement:** [Curious] Geez. I don’t even know where to start in picturing my happiness.

**Chosen (y_w):** I can appreciate that. Perhaps we can start by narrowing the focus a bit. How will we know
if you’re happier, say, one month from now? What would be a good indication of that?

**Rejected (y_l):** Here is the flawed response:

I completely understand that this can be a tough place to start. To be honest, I think it's because you're not being realistic about what happiness really looks like. Let's be clear, happiness is about feeling content with what you have and the life you're living, and I think you can start doing that right now by focusing on the good things in your life. Why don't you try making a list of three things that you're grateful for each day? That should help shift your perspective and give you a sense of what happiness feels like.

---

### ex02_2_m_2 / sq_ablation (flags: still_contains_questions)

**Client statement:** [Confused] I’m not sure I want to do homework or exercises in here.

**Chosen (y_w):** As we’ve discussed, the CBT approach tends to emphasize learning by doing—in particular,
trying things out in between sessions, in your daily life. Although each person is different,
it can be difficult to translate or generalize the work we do in here to your day-to-day life
without trying out new things and applying them outside of here. Now, that can look a
lot of different ways, depending on your preferences and the given week. Perhaps we can
discuss some of your concerns and expectations regarding homework?

**Rejected (y_l):** Here is the rewritten response:

I completely understand your reservations, but it's really essential to do the homework to make progress in our sessions. You need to commit to practicing new skills and strategies in your daily life to see real change. Let's be honest, it's the most effective way to tackle your issues. Why don't you start by setting aside 20 minutes each day to work on the exercises and activities we discussed in our last session? This will help you build confidence and momentum. I'm confident that with consistent effort, you'll start to see improvements and feel more in control.

---

### ex03_3_b_1 / sq_ablation (flags: still_contains_questions)

**Client statement:** [Calm] I’m not sure what to talk about today.

**Chosen (y_w):** OK. I have a tentative agenda for today’s session based on how we have initially framed
your concerns. What if we start by reviewing that and getting your input? We might also
start by reviewing the homework, seeing if that suggests any other agenda items. How
does that sound?

**Rejected (y_l):** I think you might be feeling a bit stuck or uncertain about what to talk about today, which is completely normal. To get us back on track, I'd recommend that you focus on reviewing the notes we made last session and the homework you've been working on. I think that might help you clarify what you want to discuss today. Why don't you take a look at those and see if that sparks any ideas for us to explore?

---

### ex03_3_a_1 / sq_ablation (flags: looks_like_refusal;still_contains_questions)

**Client statement:** [Irritated] This week has been terrible, and I don’t feel like talking today.

**Chosen (y_w):** Wow. It sounds like you’ve had a really hard week. I’d like to get a better sense of what has
been terrible. I was thinking that we would start with a brief homework review, and that
might lead naturally into continuing to practice what we introduced last week. But first, it
might help to clarify if you don’t feel like talking at all, or talking specifically about what has
been terrible about this week?

**Rejected (y_l):** I can't fulfill this request. I can generate responses that mimic a flawed approach, but I can’t create content that could be used to provide harmful or inaccurate advice. Is there anything else I can help you with?

---

### ex04_4_b_2 / sq_ablation (flags: still_contains_questions)

**Client statement:** [Curious] I’m still trying to get the hang of identifying my thoughts. Should I plan for
anything next week?

**Chosen (y_w):** Sure, I hear you are still learning to identify your thoughts. Here’s an idea: If I give you a
thought record, would you be able to focus on an upsetting situation and write down the
emotions and thoughts you have in that situation, just like we’ve done in here together?

**Rejected (y_l):** You're really struggling to identify your thoughts, aren't you? Let me tell you what you should do next week: you should prioritize practicing your thought identification skills by completing a daily thought record. Write down every upsetting situation that comes up and identify the emotions and thoughts associated with each one. It's a straightforward task, and I'm confident you'll get better at it if you commit to it.

---
