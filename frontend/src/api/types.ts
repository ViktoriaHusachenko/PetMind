export type User = {
  id: string;
  external_id: string | null;
  display_name: string | null;
  is_admin: boolean;
};

export type Pet = {
  id: string;
  user_id: string;
  name: string;
  species: string;
  birth_date: string | null;
  breed: string | null;
  weight_kg: number | null;
  allergens: string[];
  diet_restrictions: string[];
  lifestyle: string | null;
  photo_url: string | null;
};

export type PetInput = Omit<Pet, "id" | "user_id">;

export type Product = {
  id: string;
  title: string;
  brand: string | null;
  category: string | null;
  species: string | null;
  price: number | null;
  image_url: string | null;
  safe: boolean | null;
  danger_reason: string | null;
};

export type IngredientAnalysis = {
  position: number;
  name: string;
  canonical_name: string;
  allergen_group: string | null;
  is_allergen: boolean;
};

export type ProductDetail = Product & {
  description: string | null;
  ingredients_text: string | null;
  ingredients: IngredientAnalysis[];
  S_nutr: number;
  Q_ingr: number;
  S_comp: number;
  nutrient_data_missing: boolean;
  rag_chunks: Array<{ id: string; text: string; score: number }>;
};

export type RecommendationItem = {
  item_id: string;
  title: string;
  brand: string;
  price: number | null;
  image_url: string | null;
  match_percent: number;
  reason: string;
  safe_badge: boolean;
  composition_summary: string[];
  llm_used: boolean;
};

export type ReminderItem = {
  item_id: string;
  title: string;
  days_left: number;
  expected_date: string;
  text: string;
  severity: "скоро" | "сьогодні" | "прострочено";
  status: "new" | "done" | "dismissed";
  priority: number;
};

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

export type ChatResponse = {
  reply: string;
  products: RecommendationItem[];
  tool_trace: Array<Record<string, unknown>>;
};

export type ProductPage = {
  items: Product[];
  total: number;
  page: number;
  page_size: number;
};

export type Stats = {
  characteristics_count: number;
  composition_product_count: number;
};

export type ApiState<T> = {
  data: T | null;
  loading: boolean;
  error: string | null;
};

export type AuthResponse = { user: User; access_token: string };

export type AllergenOption = { value: string; label: string };

export type LLMHealth = {
  provider: string;
  state: "ok" | "disabled" | "not_configured" | "quota" | "unavailable";
  models: Record<string, string>;
  detail: string;
};
