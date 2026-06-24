#!/usr/bin/env python3
"""合并 expand 数据到 v2 + 继续扩标到 300"""
import json, os, re, time, random
from pathlib import Path
from openai import OpenAI

BASE = Path(r"D:\data\1")
EXPAND = BASE / "datasets/annotations/v2-expand"
PROCESSED = BASE / "datasets/processed"
ANNOTATED = BASE / "datasets/annotations"
V2 = BASE / "datasets/training/v2"

client = OpenAI(api_key="sk-e5h7yPSNdZg8W0HP__ZesA", base_url="https://copilot.huya.info/api/openai/v1/")
MODEL = "deepseek/deepseek-v4-pro"

P_SYS = "你是一个中文小说叙事单元分类助手。输入已切好的叙事单元。只标注类型(dialogue/narration/thought/action/scene_description)。只输出JSON: {\"labels\":[{\"unit_id\":\"...\",\"type\":\"...\"}]}"
S_SYS = "你是一个中文小说 scene 边界检测助手。只输出 boundaries。JSON: {\"boundaries\":[3]}"
A_SYS = "你是一个中文小说归因辅助助手。选最佳角色并判断不确定。只输出JSON: {\"best_candidate\":\"...\",\"uncertain\":true/false}"

def dedup(data):
    seen, result = set(), []
    for item in data:
        key = tuple(m["content"] for m in item["messages"] if m["role"] == "assistant")
        if key not in seen:
            seen.add(key); result.append(item)
    return result

def save_v2(all_data):
    random.seed(42)
    counts = {}
    for task, data in all_data.items():
        data = dedup(data)
        random.shuffle(data)
        n = len(data)
        t_end = int(n * 0.8); v_end = int(n * 0.9)
        out = V2 / task; out.mkdir(parents=True, exist_ok=True)
        for split, d in [("train", data[:t_end]), ("val", data[t_end:v_end]), ("test", data[v_end:])]:
            with open(out / f"{split}.jsonl", "w", encoding="utf-8") as fh:
                for item in d:
                    fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        counts[task] = n
    return counts

def add_sample(all_data, task, text, annotation):
    units = []
    for j, u in enumerate(re.split(r'(?<=[。！？])', text)):
        u = u.strip()
        if u and len(u) >= 2:
            units.append({"unit_id": str(j+1), "text": u})
    paragraphs = [l.strip() for l in text.split("\n") if len(l.strip()) >= 5]

    if task == "parsing" and len(units) >= 5:
        ut = "\n".join(f'[{u["unit_id"]}] {u["text"]}' for u in units)
        all_data["narrative-type-classification"].append({"messages": [
            {"role": "system", "content": P_SYS},
            {"role": "user", "content": "units:\n" + ut},
            {"role": "assistant", "content": json.dumps(annotation, ensure_ascii=False)},
        ]})
        return True
    elif task == "scene" and len(paragraphs) >= 6:
        pt = "\n".join(f"[P{j+1}] {p}" for j, p in enumerate(paragraphs[:15]))
        all_data["scene-boundary-detection"].append({"messages": [
            {"role": "system", "content": S_SYS},
            {"role": "user", "content": "段落:\n" + pt},
            {"role": "assistant", "content": json.dumps(annotation, ensure_ascii=False)},
        ]})
        return True
    elif task == "attr" and "best_candidate" in annotation:
        cs = re.findall(r'([^，。！？\n\s]{2,6})[说道回答笑喊叫嚷吼诉嘀咕怒斥低声劝问](?:[:：]|$|\s|\n)', text[:2000])
        dm = list(re.finditer(r'[“”「」]([^“”「」]{6,})[“”「」]', text))
        if cs and dm:
            cands = list(set(cs))[:8]
            target = dm[0].group(1)
            pos = dm[0].start()
            ctx = text[max(0,pos-150):min(len(text),pos+len(target)+150)].replace(target, target+"[target]")
            ct = "\n".join(f"- {c}" for c in cands)
            all_data["attribution-best-candidate"].append({"messages": [
                {"role": "system", "content": A_SYS},
                {"role": "user", "content": f"候选:\n{ct}\n\n上下文:\n{ctx}"},
                {"role": "assistant", "content": json.dumps(annotation, ensure_ascii=False)},
            ]})
            return True
    return False

# ── Load existing v2 + expand ──
all_data = {"narrative-type-classification": [], "scene-boundary-detection": [], "attribution-best-candidate": []}
for task, data in all_data.items():
    for split in ["train", "val", "test"]:
        f = V2 / task / f"{split}.jsonl"
        if f.exists():
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    data.append(json.loads(line))
print(f"Loaded v2: P={len(all_data['narrative-type-classification'])} S={len(all_data['scene-boundary-detection'])} A={len(all_data['attribution-best-candidate'])}")

added = 0
for f in sorted(EXPAND.glob("*.json")):
    if f.name == "stats.json": continue
    d = json.loads(f.read_text(encoding="utf-8"))
    fstem = f.stem
    parts = fstem.rsplit("_", 1)
    if len(parts) < 2: continue
    book_ch, task_label = parts[0], parts[1]
    m = re.match(r"(book_\d+)_(chapter_\d+)", book_ch)
    if not m: continue
    book_id, ch_id = m.group(1), m.group(2)
    ch_path = PROCESSED / book_id / f"{ch_id}.json"
    if not ch_path.exists(): continue
    ch_data = json.loads(ch_path.read_text(encoding="utf-8"))
    text = ch_data.get("chapter_text", "")[:1200]

    task_map = {"parsing": "parsing", "scene": "scene", "attr": "attr"}
    task_name = task_map.get(task_label)
    if task_name and add_sample(all_data, task_name, text, d):
        added += 1

counts = save_v2(all_data)
print(f"After merge: P={counts['narrative-type-classification']} S={counts['scene-boundary-detection']} A={counts['attribution-best-candidate']}")
print(f"Added from expand: {added}")

# ── Continue expanding ──
need = {
    "parsing": max(0, 300 - counts["narrative-type-classification"]),
    "scene": max(0, 300 - counts["scene-boundary-detection"]),
    "attribution": max(0, 200 - counts["attribution-best-candidate"]),  # attribution target 200
}
print(f"\nNeed: P={need['parsing']} S={need['scene']} A={need['attribution']}")

done_books = set()
for d in [ANNOTATED / "narrative-parsing", EXPAND]:
    if d.exists():
        for f in d.glob("*.json"):
            m = re.match(r"(book_\d+)", f.name)
            if m: done_books.add(m.group(1))

book_idx = json.loads((PROCESSED / "book_index.json").read_text(encoding="utf-8"))
avail = [b for b in book_idx if b["book_id"] not in done_books]
random.seed(123); random.shuffle(avail)

total_needed = need["parsing"] + need["scene"]
books_to_take = min(len(avail), max(20, total_needed + 10))
selected = avail[:books_to_take]
print(f"Processing {len(selected)} books")

stats = {"parsing": 0, "scene": 0, "attr": 0}
frag_count = 0

for bi, entry in enumerate(selected):
    if all(need[k] <= stats[k] for k in need if need[k] > 0):
        print(f"All targets reached!")
        break

    book_id = entry["book_id"]
    chapters = [c for c in entry.get("chapters", []) if c.get("char_count", 0) >= 800]

    for ch_info in chapters[:2]:
        ch_path = PROCESSED / book_id / f"{ch_info['chapter_id']}.json"
        if not ch_path.exists(): continue
        ch_data = json.loads(ch_path.read_text(encoding="utf-8"))
        text = ch_data.get("chapter_text", "")[:1200]
        if len(text) < 300: continue

        # Parsing
        if need["parsing"] > stats["parsing"]:
            try:
                units = []
                for j, u in enumerate(re.split(r'(?<=[。！？])', text)):
                    u = u.strip()
                    if u and len(u) >= 2:
                        units.append({"unit_id": str(j+1), "text": u})
                if len(units) >= 5:
                    ut = "\n".join(f'[{u["unit_id"]}] {u["text"]}' for u in units)
                    r = client.chat.completions.create(model=MODEL, messages=[
                        {"role": "system", "content": P_SYS},
                        {"role": "user", "content": "units:\n" + ut}],
                        max_tokens=4096, temperature=0.1, top_p=0.95)
                    raw = r.choices[0].message.content
                    m = re.search(r'\{.*\}', str(raw), re.DOTALL)
                    if m:
                        p = json.loads(m.group())
                        if "labels" in p:
                            if add_sample(all_data, "parsing", text, p):
                                stats["parsing"] += 1
                                fname = EXPAND / f"{book_id}_{ch_info['chapter_id']}_parsing.json"
                                fname.write_text(json.dumps(p, ensure_ascii=False), encoding="utf-8")
            except Exception as e:
                pass
        time.sleep(0.5)

        # Scene
        if need["scene"] > stats["scene"]:
            try:
                paragraphs = [l.strip() for l in text.split("\n") if len(l.strip()) >= 5]
                if len(paragraphs) >= 6:
                    pt = "\n".join(f"[P{j+1}] {p}" for j, p in enumerate(paragraphs[:15]))
                    r = client.chat.completions.create(model=MODEL, messages=[
                        {"role": "system", "content": S_SYS},
                        {"role": "user", "content": "段落:\n" + pt}],
                        max_tokens=4096, temperature=0.1, top_p=0.95)
                    raw = r.choices[0].message.content
                    m = re.search(r'\{.*\}', str(raw), re.DOTALL)
                    if m:
                        s = json.loads(m.group())
                        if "boundaries" in s:
                            if add_sample(all_data, "scene", text, s):
                                stats["scene"] += 1
                                fname = EXPAND / f"{book_id}_{ch_info['chapter_id']}_scene.json"
                                fname.write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
            except Exception as e:
                pass
        time.sleep(0.5)

        # Attribution
        if need["attribution"] > stats["attr"]:
            try:
                cs = re.findall(r'([^，。！？\n\s]{2,6})[说道回答笑喊叫嚷吼诉嘀咕怒斥低声劝问](?:[:：]|$|\s|\n)', text[:2000])
                dm = list(re.finditer(r'[“”「」]([^“”「」]{6,})[“”「」]', text))
                if cs and dm:
                    cands = list(set(cs))[:8]
                    target = dm[0].group(1)
                    pos = dm[0].start()
                    ctx = text[max(0,pos-150):min(len(text),pos+len(target)+150)].replace(target, target+"[target]")
                    ct = "\n".join(f"- {c}" for c in cands)
                    r = client.chat.completions.create(model=MODEL, messages=[
                        {"role": "system", "content": A_SYS},
                        {"role": "user", "content": "候选:\n" + ct + "\n\n上下文:\n" + ctx}],
                        max_tokens=4096, temperature=0.1, top_p=0.95)
                    raw = r.choices[0].message.content
                    m = re.search(r'\{.*\}', str(raw), re.DOTALL)
                    if m:
                        a = json.loads(m.group())
                        if "best_candidate" in a and "uncertain" in a:
                            if add_sample(all_data, "attr", text, a):
                                stats["attr"] += 1
                                fname = EXPAND / f"{book_id}_{ch_info['chapter_id']}_attr.json"
                                fname.write_text(json.dumps(a, ensure_ascii=False), encoding="utf-8")
            except Exception as e:
                pass
        time.sleep(0.5)

        frag_count += 1
        if frag_count % 30 == 0:
            print(f"  [{frag_count}] P+{stats['parsing']} S+{stats['scene']} A+{stats['attr']}")

# Final save
print(f"\nExpansion: {stats}")
final = save_v2(all_data)
print(f"Final: P={final['narrative-type-classification']} S={final['scene-boundary-detection']} A={final['attribution-best-candidate']}")
