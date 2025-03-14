import re
from typing import Callable, Union


def _postprocess_time_interval(time_interval: str) -> str:
    time_interval = time_interval.strip().lower()
    time_interval = time_interval.replace("**", "")
    time_interval = time_interval.replace("few", "3")
    time_interval = time_interval.replace("several", "3")
    time_interval = time_interval.replace("$\boxed{", "").replace("}$", "")
    time_interval = time_interval.replace("\\sqrt{", "").replace("}", "")
    time_interval = time_interval.replace("half an", "0.5")
    time_interval = time_interval.replace("half a", "0.5")
    time_interval = time_interval.replace("a half", "0.5")
    return time_interval


def convert_to_minutes(time_interval: str) -> float:
    time_interval = _postprocess_time_interval(time_interval)
    time_pattern = re.compile(
        r"(\d*\.?\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)\s*(seconds?|minutes?|hours?|days?|weeks?)",
        re.IGNORECASE,
    )
    range_pattern = re.compile(r"(\d+)-(\d+)\s*(minutes?|hours?)", re.IGNORECASE)
    iso_pattern = re.compile(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", re.IGNORECASE)

    number_words = {
        "a": 1,
        "an": 1,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
        "hundred": 100,
        "thousand": 1000,
    }

    match_range = range_pattern.search(time_interval)
    if match_range:
        start, end, unit = match_range.groups()
        start = float(start)
        end = float(end)
        avg = (start + end) / 2
        unit = unit.lower()

        conversion = {
            "second": avg / 60,
            "seconds": avg / 60,
            "minute": avg,
            "minutes": avg,
            "hour": avg * 60,
            "hours": avg * 60,
            "day": avg * 1440,
            "days": avg * 1440,
            "week": avg * 10080,
            "weeks": avg * 10080,
        }

        return conversion.get(unit, 0.0)

    match = time_pattern.search(time_interval)
    if match:
        value, unit = match.groups()
        value = float(number_words.get(value.lower(), value))
        unit = unit.lower()

        conversion = {
            "second": value / 60,
            "seconds": value / 60,
            "minute": value,
            "minutes": value,
            "hour": value * 60,
            "hours": value * 60,
            "day": value * 1440,
            "days": value * 1440,
            "week": value * 10080,
            "weeks": value * 10080,
        }

        return conversion.get(unit, 0.0)

    match_iso = iso_pattern.search(time_interval)
    if match_iso:
        days, hours, minutes, seconds = match_iso.groups()
        total_minutes = 0

        if days:
            total_minutes += int(days) * 1440
        if hours:
            total_minutes += int(hours) * 60
        if minutes:
            total_minutes += int(minutes)
        if seconds:
            total_minutes += int(seconds) / 60
        return total_minutes

    return 0.0


def convert_to_minutes_cot(time_interval: str) -> float:
    answer_pattern = re.compile(
        r"Therefore, the answer:\s*(\d*\.?\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)\s*(seconds?|minutes?|hours?|days?|weeks?)",
        re.IGNORECASE,
    )
    other_pattern = re.compile(
        r"(\d*\.?\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)\s*(seconds?|minutes?|hours?|days?|weeks?)",
        re.IGNORECASE,
    )

    number_words = {
        "a": 1,
        "an": 1,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
        "hundred": 100,
        "thousand": 1000,
    }

    match = answer_pattern.search(time_interval)
    if match:
        value, unit = match.groups()
        value = float(number_words.get(value.lower(), value))
        unit = unit.lower()

        conversion = {
            "second": value / 60,
            "seconds": value / 60,
            "minute": value,
            "minutes": value,
            "hour": value * 60,
            "hours": value * 60,
            "day": value * 1440,
            "days": value * 1440,
            "week": value * 10080,
            "weeks": value * 10080,
        }

        return conversion.get(unit, 0.0)

    match = other_pattern.search(time_interval)
    if match:
        value, unit = match.groups()
        value = float(number_words.get(value.lower(), value))
        unit = unit.lower()

        conversion = {
            "second": value / 60,
            "seconds": value / 60,
            "minute": value,
            "minutes": value,
            "hour": value * 60,
            "hours": value * 60,
            "day": value * 1440,
            "days": value * 1440,
            "week": value * 10080,
            "weeks": value * 10080,
        }

        return conversion.get(unit, 0.0)

    return 0.0


def postprocess_response(response: str) -> str:
    response = response.strip()
    response = re.sub(r"^[A-Z]:\s*\(.*?\)", "", response).strip()
    response = re.sub(r"[ ]+", " ", response).strip()
    response = response.split("\n")[0].strip()
    response = re.sub(r"^[A-Z]\s*:", "", response).strip()
    for _ in range(3):
        response = re.sub(r"^[\[\(].*?[\]\)]", "", response).strip()
    response = response.split("A:")[0].strip()
    response = response.split("B:")[0].strip()
    response = response.split("#")[0].strip()
    response = response.split("``")[0].strip()
    response = response.split("--")[0].strip()
    response = response.split("==")[0].strip()
    response = response.split("__")[0].strip()
    response = response.split("''")[0].strip()
    response = response.split("[")[0].strip()
    response = response.split("]")[0].strip()
    response = response.split("<")[0].strip()
    response = response.split(">")[0].strip()
    response = response.split("(")[0].strip()
    response = response.split(")")[0].strip()
    response = response.split(" ... ")[0].strip()
    response = re.sub(r"\(.*?\)", "", response).strip()
    response = re.sub(r"\[.*?\]", "", response).strip()
    return response


def postprocess_response_cot(response: str) -> str:
    response = response.strip()
    answer_pattern = re.compile(r"Therefore, the answer:\s*(.+)", re.IGNORECASE)
    match = answer_pattern.search(response)
    if match:
        response = match.group(1).strip()
        response = re.sub(r"^[A-Z]:\s*", "", response).strip()
        response = re.sub(r"[ ]+", " ", response).strip()
        response = response.split("A:")[0].strip()
        response = response.split("B:")[0].strip()
        response = response.split("#")[0].strip()
        response = response.split("``")[0].strip()
        response = response.split("--")[0].strip()
        response = response.split("==")[0].strip()
        response = response.split("__")[0].strip()
        response = response.split("''")[0].strip()
        return response
    other_pattern = re.compile(r"[A-Z]:\s*(.+)", re.IGNORECASE)
    match = other_pattern.search(response)
    if match:
        response = match.group(1).strip()
        response = re.sub(r"^[A-Z]:\s*", "", response).strip()
        response = re.sub(r"[ ]+", " ", response).strip()
        response = response.split("A:")[0].strip()
        response = response.split("B:")[0].strip()
        response = response.split("#")[0].strip()
        response = response.split("``")[0].strip()
        response = response.split("--")[0].strip()
        response = response.split("==")[0].strip()
        response = response.split("__")[0].strip()
        response = response.split("''")[0].strip()
        return response
    return response


def get_postprocess_fn(task: str, icl_method: str) -> Callable[[str], Union[str, float]]:
    if task == "time":
        if icl_method == "cot":
            return convert_to_minutes_cot
        else:
            return convert_to_minutes
    elif task == "response":
        if icl_method == "cot":
            return postprocess_response_cot
        else:
            return postprocess_response
    else:
        raise ValueError(f"Invalid task: {task}")
