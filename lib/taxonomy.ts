// Maps the raw `category` values in the dataset onto a taxonomic class and
// an order-level group, for the landing-page statistics. Values not listed
// here (and null) count as "not recorded".

export const CLASSES = [
  "Birds",
  "Mammals",
  "Reptiles",
  "Fish",
  "Amphibians",
  "Invertebrates",
] as const;
export type TaxonClass = (typeof CLASSES)[number];

interface TaxonInfo {
  cls: TaxonClass;
  order: string;
}

const M: Record<string, TaxonInfo> = {
  // Birds
  "Avian, Psittacine": { cls: "Birds", order: "Psittaciformes (parrots)" },
  "Avian, Passerine": { cls: "Birds", order: "Passeriformes" },
  "Avian, Corvidae": { cls: "Birds", order: "Passeriformes" },
  "Avian, Anseriformes": { cls: "Birds", order: "Anseriformes (waterfowl)" },
  "Avian, Columbiformes": { cls: "Birds", order: "Columbiformes (pigeons & doves)" },
  "Avian, Galliforme": { cls: "Birds", order: "Galliformes" },
  "Avian, Phasianidae": { cls: "Birds", order: "Galliformes" },
  "Avian, Raptor": { cls: "Birds", order: "Raptors" },
  "Avian, Ciconiiformes": { cls: "Birds", order: "Ciconiiformes" },
  "Avian, Ramphastids": { cls: "Birds", order: "Piciformes (toucans, woodpeckers)" },
  "Avian, Piciformes": { cls: "Birds", order: "Piciformes (toucans, woodpeckers)" },
  "Avian, Charadriiformes": { cls: "Birds", order: "Charadriiformes" },
  "Avian, Coraciiformes": { cls: "Birds", order: "Coraciiformes" },
  "Avian, Ratites": { cls: "Birds", order: "Struthioniformes (ratites)" },
  "Avian, Gruiformes": { cls: "Birds", order: "Gruiformes" },
  "Avian, Sphenisciformes": { cls: "Birds", order: "Sphenisciformes (penguins)" },
  "Avian, Pelecaniformes": { cls: "Birds", order: "Pelecaniformes" },
  "Avian, Cuculiformes": { cls: "Birds", order: "Cuculiformes" },
  "Avian, Coliiformes": { cls: "Birds", order: "Coliiformes (mousebirds)" },
  "Avian, Apodiformes": { cls: "Birds", order: "Apodiformes" },
  "Avian, Gaviiformes": { cls: "Birds", order: "Gaviiformes (loons)" },
  "Avian, Caprimulgiformes": { cls: "Birds", order: "Caprimulgiformes" },
  "Avian, Procellariiformes": { cls: "Birds", order: "Procellariiformes" },
  "Avian, Trogoniformes": { cls: "Birds", order: "Trogoniformes" },
  "Avian, Podicipedidae": { cls: "Birds", order: "Podicipediformes (grebes)" },
  "Avian, Suliformes": { cls: "Birds", order: "Suliformes" },
  Avian: { cls: "Birds", order: "Unspecified birds" },

  // Mammals
  Mustelidae: { cls: "Mammals", order: "Carnivora (ferrets & relatives)" },
  Feline: { cls: "Mammals", order: "Carnivora (cats)" },
  Canine: { cls: "Mammals", order: "Carnivora (dogs)" },
  Procyonidae: { cls: "Mammals", order: "Carnivora (other)" },
  Viverridae: { cls: "Mammals", order: "Carnivora (other)" },
  Ursulid: { cls: "Mammals", order: "Carnivora (other)" },
  Ailuridae: { cls: "Mammals", order: "Carnivora (other)" },
  Hyaenidae: { cls: "Mammals", order: "Carnivora (other)" },
  Meerkat: { cls: "Mammals", order: "Carnivora (other)" },
  Rodent: { cls: "Mammals", order: "Rodentia" },
  Rodentia: { cls: "Mammals", order: "Rodentia" },
  Lagomorph: { cls: "Mammals", order: "Lagomorpha (rabbits)" },
  Insectivora: { cls: "Mammals", order: "Insectivores (hedgehogs & allies)" },
  Insectivore: { cls: "Mammals", order: "Insectivores (hedgehogs & allies)" },
  Marsupial: { cls: "Mammals", order: "Marsupials" },
  "Primate, New World": { cls: "Mammals", order: "Primates" },
  "Primate, Apes": { cls: "Mammals", order: "Primates" },
  "Primate, Old World": { cls: "Mammals", order: "Primates" },
  "Primate, Prosimian": { cls: "Mammals", order: "Primates" },
  "Primate, Lorisidae": { cls: "Mammals", order: "Primates" },
  Primate: { cls: "Mammals", order: "Primates" },
  Chiroptera: { cls: "Mammals", order: "Chiroptera (bats)" },
  Antelope: { cls: "Mammals", order: "Artiodactyla (even-toed ungulates)" },
  Caprine: { cls: "Mammals", order: "Artiodactyla (even-toed ungulates)" },
  Ovine: { cls: "Mammals", order: "Artiodactyla (even-toed ungulates)" },
  Bovine: { cls: "Mammals", order: "Artiodactyla (even-toed ungulates)" },
  Cervid: { cls: "Mammals", order: "Artiodactyla (even-toed ungulates)" },
  Ruminant: { cls: "Mammals", order: "Artiodactyla (even-toed ungulates)" },
  "Ruminant, Giraffidae": { cls: "Mammals", order: "Artiodactyla (even-toed ungulates)" },
  "Ruminant, Tragulidae": { cls: "Mammals", order: "Artiodactyla (even-toed ungulates)" },
  Ungulants: { cls: "Mammals", order: "Artiodactyla (even-toed ungulates)" },
  Artiodactyla: { cls: "Mammals", order: "Artiodactyla (even-toed ungulates)" },
  Porcine: { cls: "Mammals", order: "Artiodactyla (even-toed ungulates)" },
  Camilid: { cls: "Mammals", order: "Artiodactyla (even-toed ungulates)" },
  Equine: { cls: "Mammals", order: "Perissodactyla" },
  Tapiroidea: { cls: "Mammals", order: "Perissodactyla" },
  "Marine Mammal": { cls: "Mammals", order: "Marine mammals" },
  Edentata: { cls: "Mammals", order: "Xenarthra" },
  Hyracoidea: { cls: "Mammals", order: "Hyracoidea (hyraxes)" },
  Orycteropodidae: { cls: "Mammals", order: "Tubulidentata (aardvark)" },
  Elephant: { cls: "Mammals", order: "Proboscidea (elephants)" },
  Scandentia: { cls: "Mammals", order: "Scandentia (treeshrews)" },
  "Mammal-Not Provided": { cls: "Mammals", order: "Unspecified mammals" },
  Mammal: { cls: "Mammals", order: "Unspecified mammals" },
  "Mammal?": { cls: "Mammals", order: "Unspecified mammals" },

  // Reptiles
  "Reptile, Lizard": { cls: "Reptiles", order: "Lizards (Sauria)" },
  "Reptile, Snake": { cls: "Reptiles", order: "Snakes (Serpentes)" },
  "Reptile, Chelonian": { cls: "Reptiles", order: "Chelonians (Testudines)" },
  "Reptile, Crocodilian": { cls: "Reptiles", order: "Crocodilians" },
  Reptile: { cls: "Reptiles", order: "Unspecified reptiles" },

  // Amphibians
  "Amphibian, anuran": { cls: "Amphibians", order: "Anura (frogs & toads)" },
  "Amphibian, Anurian": { cls: "Amphibians", order: "Anura (frogs & toads)" },
  "Amphibian, anurian": { cls: "Amphibians", order: "Anura (frogs & toads)" },
  "Amphibian, Caudata": { cls: "Amphibians", order: "Caudata (salamanders)" },

  // Fish
  Fish: { cls: "Fish", order: "Bony fish" },
  "Fish, Sharks and Rays": { cls: "Fish", order: "Sharks & rays" },

  // Invertebrates
  "Cnidaria Phylum": { cls: "Invertebrates", order: "Cnidarians" },
  Bugs: { cls: "Invertebrates", order: "Arthropods" },
  Echinoderm: { cls: "Invertebrates", order: "Echinoderms" },
};

export function classify(category: string | null): TaxonInfo | null {
  if (!category) return null;
  return M[category] ?? null;
}
