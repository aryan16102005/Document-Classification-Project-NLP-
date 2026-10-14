"""
Sample document generator — creates realistic training data for 8 categories.
src/data/generator.py
"""

import random
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from typing import List, Dict, Optional


class SampleDataGenerator:
    """
    Generates synthetic but realistic document samples for each category.
    Used to bootstrap the system without requiring an external dataset.
    """

    TEMPLATES: Dict[str, List[str]] = {
        "technology": [
            "Apple unveiled its latest {product} featuring revolutionary {feature} that promises to {benefit}. "
            "The new {product} uses advanced {tech} to deliver unprecedented performance. "
            "Tech analysts say this could {impact} the entire industry.",
            "Google DeepMind researchers published a breakthrough paper on {ai_topic}, demonstrating "
            "that neural networks can {capability}. The study, conducted over {period}, shows "
            "a {percentage}% improvement over existing methods.",
            "Microsoft announced a {amount} billion dollar investment in {tech_area}, aiming to {goal}. "
            "The company plans to deploy {number} new data centers across {region} by {year}.",
            "A new {programming_language} framework called {name} has gained massive adoption among developers. "
            "It promises to simplify {task} with just {lines} lines of code.",
            "Cybersecurity researchers discovered a critical vulnerability in {software} affecting "
            "millions of users worldwide. The {company} has issued an emergency patch.",
        ],
        "sports": [
            "{team} defeated {opponent} in a thrilling {sport} match, with {player} scoring "
            "{score} points in the final quarter. The victory puts {team} at the top of {league}.",
            "The {country} national {sport} team secured a historic {medal} medal at the "
            "{tournament} championship after defeating {rival} by {margin}.",
            "{player}, widely regarded as the greatest {sport} player of all time, announced "
            "retirement after {years} years. The {nationality} athlete holds {records} world records.",
            "The {year} {tournament} saw record-breaking attendance of {number} fans. "
            "{team} emerged as champions after beating {opponent} in the finals.",
            "Transfer market: {player} moves from {club1} to {club2} for a record fee of "
            "{amount} million euros, becoming the most expensive {position} in history.",
        ],
        "politics": [
            "The government passed the {bill_name} Act, which aims to {goal}. "
            "Opposition lawmakers criticized the bill, arguing it would {negative_effect}. "
            "The legislation is expected to affect {number} million citizens.",
            "President {name} signed an executive order on {topic}, calling it a {descriptor} "
            "step toward {goal}. The order takes effect immediately.",
            "The {country} parliament voted {for} to {against} in favor of the new {policy} policy. "
            "Political analysts say this marks a significant shift in the nation's approach to {topic}.",
            "United Nations Secretary-General called for immediate action on {global_issue}, "
            "warning that without cooperation, {consequence} by {year}.",
            "Local elections in {city} resulted in a landslide victory for the {party} party, "
            "marking the first time in {years} years they have controlled {position}.",
        ],
        "entertainment": [
            "{movie} smashed box office records, earning {amount} million in its opening weekend. "
            "Directed by {director}, the film stars {actor} in a career-defining performance.",
            "{singer} released their highly anticipated album '{album_title}', which broke "
            "streaming records with {number} million listens in its first {period}.",
            "The {award} Awards ceremony crowned '{title}' as the best {category} of the year. "
            "{celebrity} delivered an emotional acceptance speech.",
            "Netflix announced a new {genre} series '{show_name}', based on the best-selling "
            "novel by {author}. Production begins in {month} with a {budget} million budget.",
            "{celebrity} and {celebrity2} confirmed their relationship after months of speculation. "
            "The couple was spotted together at the {event} premiere in {city}.",
        ],
        "science": [
            "Scientists at {institution} discovered {discovery}, potentially revolutionizing "
            "our understanding of {field}. The research was published in {journal}.",
            "NASA's {mission} spacecraft captured unprecedented images of {celestial_body}, "
            "revealing {finding}. The mission, costing {amount} billion dollars, has "
            "been in operation for {years} years.",
            "A new species of {organism} was discovered in {location}, challenging previous "
            "assumptions about {topic}. Researchers believe the species has existed for "
            "{years} million years.",
            "Climate scientists warn that global temperatures could rise by {degrees} degrees "
            "Celsius by {year} if emissions continue at current levels. The study analyzed "
            "data from {number} weather stations worldwide.",
            "Physicists at CERN confirmed the existence of {particle}, a theoretical particle "
            "that could explain {phenomenon}. The discovery required {years} years of research.",
        ],
        "health": [
            "A new study published in {journal} found that {activity} reduces the risk of "
            "{disease} by {percentage}% in adults over {age}. Researchers followed "
            "{number} participants for {years} years.",
            "The FDA approved {drug_name}, a breakthrough treatment for {disease}, which "
            "has shown {percentage}% effectiveness in clinical trials. The drug works by {mechanism}.",
            "Health officials confirmed {number} new cases of {disease} in {region}, prompting "
            "authorities to issue {measures}. Hospitals are preparing for potential surge.",
            "Researchers discovered that {food} contains a compound called {compound} that "
            "may help prevent {disease}. The finding could lead to new {treatment} therapies.",
            "Mental health experts report a {percentage}% increase in {condition} cases since "
            "{year}, urging governments to invest more in {treatment} programs.",
        ],
        "finance": [
            "The stock market {direction} sharply today as {reason}. The S&P 500 {moved} "
            "by {percentage}%, while the Nasdaq {moved2} {percentage2}%.",
            "{company} reported quarterly earnings of {amount} billion dollars, {exceeding_missing} "
            "analyst expectations by {percentage}%. Revenue grew {rate}% year-over-year.",
            "The Federal Reserve {action} interest rates by {basis_points} basis points, "
            "citing {reason}. This marks the {ordinal} rate change this year.",
            "Bitcoin {direction} to {amount} dollars as {reason}. Cryptocurrency markets "
            "saw {volume} billion dollars in trading volume over the past {period}.",
            "A merger between {company1} and {company2}, valued at {amount} billion dollars, "
            "was approved by regulators. The combined entity will employ {number} people globally.",
        ],
        "world": [
            "A powerful {magnitude} magnitude earthquake struck {region}, causing widespread "
            "damage. At least {number} people were killed and {number2} injured. "
            "Rescue teams are searching for survivors.",
            "Tensions escalated between {country1} and {country2} over {issue}. "
            "International mediators from the {organization} are working to {goal}.",
            "A historic peace agreement was signed between {party1} and {party2}, ending "
            "{years} years of conflict. The deal was brokered by {mediator}.",
            "{number} refugees fled {country} following {event}, according to {organization}. "
            "Neighboring countries have pledged {amount} million dollars in humanitarian aid.",
            "A major {disaster_type} in {region} has left {number} people homeless. "
            "International aid organizations are responding to the crisis.",
        ],
    }

    FILLERS = {
        "product": ["iPhone 17", "MacBook Pro M4", "Surface Pro X", "Galaxy S25", "Pixel 9"],
        "feature": ["AI integration", "quantum computing chip", "120Hz display", "satellite connectivity"],
        "benefit": ["redefine mobile computing", "transform productivity", "enable new use cases"],
        "tech": ["machine learning", "neural processing", "edge computing", "photonic chips"],
        "impact": ["disrupt", "transform", "reshape", "revolutionize"],
        "ai_topic": ["multi-modal reasoning", "code generation", "protein folding", "language understanding"],
        "capability": ["solve complex problems", "understand context better", "learn from fewer examples"],
        "period": ["two years", "six months", "18 months", "three years"],
        "percentage": ["23", "47", "65", "89", "12", "34"],
        "amount": ["50", "120", "3.5", "17", "85", "200"],
        "tech_area": ["AI infrastructure", "cloud computing", "quantum research", "cybersecurity"],
        "goal": ["accelerate innovation", "create jobs", "reduce costs", "improve efficiency"],
        "number": ["100", "500", "50,000", "2 million", "10"],
        "region": ["North America", "Europe", "Asia Pacific", "Southeast Asia"],
        "year": ["2025", "2026", "2027", "2030"],
        "programming_language": ["Python", "Rust", "Go", "TypeScript"],
        "name": ["Quantum", "Nexus", "Aurora", "Helios"],
        "task": ["web development", "data processing", "API design", "testing"],
        "lines": ["10", "5", "20", "50"],
        "software": ["Windows", "Android", "Chrome", "OpenSSL"],
        "company": ["Microsoft", "Google", "Apple", "Meta", "Amazon"],
        "team": ["Manchester City", "Barcelona", "LA Lakers", "New Zealand", "India"],
        "opponent": ["Real Madrid", "PSG", "Boston Celtics", "Australia", "England"],
        "sport": ["football", "basketball", "cricket", "tennis", "rugby"],
        "player": ["Messi", "LeBron James", "Virat Kohli", "Serena Williams"],
        "score": ["42", "28", "156", "7"],
        "league": ["Premier League", "La Liga", "NBA", "Champions League"],
        "country": ["Brazil", "Germany", "Japan", "France", "Spain"],
        "medal": ["gold", "silver", "bronze"],
        "tournament": ["World Cup", "Olympics", "Champions League", "Grand Slam"],
        "rival": ["Argentina", "England", "South Korea", "Italy"],
        "margin": ["3 goals", "2 points", "15 runs", "6-3"],
        "nationality": ["Brazilian", "German", "Japanese", "French"],
        "records": ["12", "8", "25", "5"],
        "club1": ["PSG", "Manchester United", "Bayern Munich", "Juventus"],
        "club2": ["Real Madrid", "Barcelona", "Manchester City", "Chelsea"],
        "position": ["striker", "midfielder", "defender", "goalkeeper"],
        "bill_name": ["Climate Action", "Digital Privacy", "Infrastructure", "Healthcare Reform"],
        "negative_effect": ["harm the economy", "infringe on civil liberties", "increase inequality"],
        "descriptor": ["historic", "bold", "unprecedented", "controversial"],
        "topic": ["climate change", "immigration", "trade", "cybersecurity", "taxation"],
        "for": ["312", "248", "189", "421"],
        "against": ["128", "192", "211", "79"],
        "policy": ["economic", "foreign", "environmental", "social", "defense"],
        "global_issue": ["climate change", "food security", "nuclear disarmament", "pandemic preparedness"],
        "consequence": ["catastrophic damage will occur", "millions will face poverty", "conflicts will escalate"],
        "city": ["Paris", "New York", "London", "Tokyo", "Berlin"],
        "party": ["Progressive", "Conservative", "Green", "Labour", "Democratic"],
        "years": ["10", "20", "50", "5", "30"],
        "position": ["city council", "the mayorship", "the state assembly"],
        "movie": ["The Last Frontier", "Echoes of Time", "Dark Horizon", "The Quantum Code"],
        "director": ["James Cameron", "Ava DuVernay", "Christopher Nolan", "Bong Joon-ho"],
        "actor": ["Timothée Chalamet", "Zendaya", "Cate Blanchett", "Mahershala Ali"],
        "singer": ["Taylor Swift", "Beyoncé", "Bad Bunny", "Drake"],
        "album_title": ["Renaissance II", "Eras Continues", "Neon Nights", "Midnight"],
        "award": ["Grammy", "Academy", "BAFTA", "Golden Globe"],
        "title": ["The Crown Season 6", "Oppenheimer II", "Everything Everywhere"],
        "category": ["drama", "comedy", "documentary", "animated film"],
        "celebrity": ["Jennifer Lopez", "Brad Pitt", "Rihanna", "Ryan Gosling"],
        "celebrity2": ["Chris Evans", "Margot Robbie", "Tom Holland", "Selena Gomez"],
        "genre": ["thriller", "sci-fi", "fantasy", "crime", "romantic"],
        "show_name": ["The Midnight Garden", "Nexus Protocol", "Rising Tides"],
        "author": ["Stephen King", "Gillian Flynn", "Colleen Hoover", "John Grisham"],
        "month": ["January", "March", "June", "September"],
        "budget": ["50", "120", "200", "75"],
        "event": ["film", "premiere", "charity gala", "awards"],
        "institution": ["MIT", "Stanford", "Oxford", "Johns Hopkins", "CERN"],
        "discovery": ["a new exoplanet", "a cure pathway", "a quantum state", "gravitational waves"],
        "field": ["cosmology", "medicine", "quantum mechanics", "biology"],
        "journal": ["Nature", "Science", "The Lancet", "NEJM", "Cell"],
        "mission": ["Artemis", "James Webb", "Voyager 3", "Europa Clipper"],
        "celestial_body": ["Mars", "Europa", "a distant galaxy", "a black hole"],
        "finding": ["evidence of water", "complex organic molecules", "unexpected magnetic fields"],
        "organism": ["deep-sea fish", "insect", "plant", "microorganism", "primate"],
        "location": ["the Amazon rainforest", "the Mariana Trench", "Siberia", "Antarctica"],
        "degrees": ["1.5", "2.0", "2.5", "3.0"],
        "particle": ["the W boson", "dark matter", "a new quark", "axion"],
        "phenomenon": ["dark energy", "galaxy formation", "cosmic inflation"],
        "activity": ["regular exercise", "Mediterranean diet", "intermittent fasting", "mindfulness"],
        "disease": ["heart disease", "type 2 diabetes", "Alzheimer's", "cancer", "depression"],
        "age": ["40", "50", "60", "65"],
        "drug_name": ["NovaCure", "AlzheimX", "CardioShield", "NeuroBoost"],
        "mechanism": ["targeting specific proteins", "blocking inflammatory pathways", "regulating gene expression"],
        "measures": ["travel restrictions", "vaccination campaigns", "quarantine protocols"],
        "food": ["blueberries", "green tea", "turmeric", "olive oil", "broccoli"],
        "compound": ["resveratrol", "curcumin", "anthocyanin", "EGCG"],
        "treatment": ["drug", "gene therapy", "immunotherapy", "cognitive behavioral"],
        "condition": ["anxiety", "depression", "PTSD", "eating disorder"],
        "direction": ["rose", "fell", "surged", "plummeted"],
        "reason": ["inflation data surprised analysts", "Fed signaled rate cuts", "earnings beat expectations"],
        "moved": ["gained", "lost", "jumped", "dropped"],
        "percentage2": ["1.8", "3.2", "0.9", "4.1"],
        "moved2": ["climbed", "fell", "surged", "retreated"],
        "exceeding_missing": ["exceeding", "missing"],
        "rate": ["12", "8", "25", "5"],
        "action": ["raised", "cut", "held steady on"],
        "basis_points": ["25", "50", "75", "100"],
        "ordinal": ["third", "second", "fourth", "fifth"],
        "volume": ["50", "120", "80", "200"],
        "company1": ["Apple", "Microsoft", "Amazon", "Google"],
        "company2": ["Disney", "Paramount", "Warner Bros", "MGM"],
        "magnitude": ["6.2", "7.1", "5.8", "7.5", "6.9"],
        "region": ["central Turkey", "southern Japan", "northern California", "Nepal", "Indonesia"],
        "number2": ["hundreds", "thousands", "1,200", "350"],
        "country1": ["USA", "China", "Russia", "India"],
        "country2": ["China", "Taiwan", "Ukraine", "Pakistan"],
        "issue": ["territorial disputes", "trade tariffs", "border security", "nuclear weapons"],
        "organization": ["UN", "NATO", "ASEAN", "African Union"],
        "party1": ["the government", "rebel forces", "opposition leaders"],
        "party2": ["militant groups", "the military", "separatist movements"],
        "mediator": ["the United Nations", "USA", "European Union"],
        "event": ["political crisis", "conflict", "natural disaster", "economic collapse"],
        "disaster_type": ["flood", "wildfire", "hurricane", "drought", "landslide"],
    }

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.categories = list(self.TEMPLATES.keys())
        random.seed(self.config.get("random_seed", 42))
        np.random.seed(self.config.get("random_seed", 42))

    def _fill_template(self, template: str) -> str:
        """Replace all {placeholder} tokens in a template string."""
        import re
        placeholders = re.findall(r'\{(\w+)\}', template)
        for ph in placeholders:
            if ph in self.FILLERS:
                template = template.replace(
                    "{" + ph + "}", random.choice(self.FILLERS[ph]), 1
                )
        return template

    def generate(self, n_per_category: int = 500) -> pd.DataFrame:
        """
        Generate a balanced dataset with n_per_category samples per class.

        Args:
            n_per_category: Number of documents per category.

        Returns:
            DataFrame with columns: [text, label, label_id]
        """
        records = []
        label_map = {cat: idx for idx, cat in enumerate(self.categories)}

        logger.info(f"Generating {n_per_category} samples per category × {len(self.categories)} categories...")

        for category, templates in self.TEMPLATES.items():
            for i in range(n_per_category):
                # Pick a random template and fill it
                template = random.choice(templates)
                text = self._fill_template(template)

                # Optionally add a second sentence from another template
                if random.random() > 0.5:
                    extra = self._fill_template(random.choice(templates))
                    text = text + " " + extra

                records.append({
                    "text": text.strip(),
                    "label": category,
                    "label_id": label_map[category],
                })

        df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info(f"Generated {len(df)} total samples.")
        return df

    def save(self, df: pd.DataFrame, output_dir: str = "data/sample") -> None:
        """Save the generated dataset as CSV files (train/val/test splits)."""
        from sklearn.model_selection import train_test_split

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Full dataset
        full_path = output_path / "full_dataset.csv"
        df.to_csv(full_path, index=False)
        logger.info(f"Saved full dataset → {full_path}")

        # Train / val / test splits (70/15/15)
        train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
        val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=42)

        for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
            path = output_path / f"{split_name}.csv"
            split_df.to_csv(path, index=False)
            logger.info(f"Saved {split_name} split ({len(split_df)} rows) → {path}")

        logger.success(f"Dataset splits: train={len(train)}, val={len(val)}, test={len(test)}")
