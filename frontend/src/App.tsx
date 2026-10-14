import { useEffect, useState } from "react";
import { Link, Navigate, NavLink, Route, Routes, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { apiRequest, useApiResource } from "./api/client";
import type { AuthResponse, ChatMessage, ChatResponse, Pet, PetInput, Product, ProductDetail, ProductPage, RecommendationItem, ReminderItem, Stats, User } from "./api/types";

type IconName = "search" | "user" | "heart" | "cart" | "menu" | "spark" | "arrow" | "check" | "shield" | "truck" | "box" | "refresh" | "close" | "plus" | "send" | "paw";
type StoreProduct = { id?: string; item_id?: string; title: string; brand?: string | null; category?: string | null; species?: string | null; price?: number | null; image_url?: string | null; safe?: boolean | null; danger_reason?: string | null; match_percent?: number; safe_badge?: boolean; reason?: string; composition_summary?: string[]; cart_pet_id?: string };
type CartLine = { product: StoreProduct; pet_id: string };
function productId(product: StoreProduct): string { return product.item_id || product.id || ""; }
function formatDateUk(isoDate: string): string {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(isoDate);
  return match ? `${match[3]}.${match[2]}.${match[1]}` : "";
}
function parseDateUk(value: string): string | null {
  const match = /^(\d{2})\.(\d{2})\.(\d{4})$/.exec(value.trim());
  if (!match) return null;
  const [, dayText, monthText, yearText] = match;
  const day = Number(dayText), month = Number(monthText), year = Number(yearText);
  const date = new Date(Date.UTC(year, month - 1, day));
  if (date.getUTCFullYear() !== year || date.getUTCMonth() !== month - 1 || date.getUTCDate() !== day) return null;
  return `${yearText}-${monthText}-${dayText}`;
}
function readSavedAuthUser(): User | null {
  if (!localStorage.getItem("petmind_access_token")) return null;
  try { return JSON.parse(localStorage.getItem("petmind_auth_user") || "null") as User | null; }
  catch { return null; }
}

function Icon({ name, size = 20 }: { name: IconName; size?: number }) {
  const paths: Record<IconName, React.ReactNode> = {
    search: <><circle cx="11" cy="11" r="7" /><path d="m20 20-4-4" /></>, user: <><circle cx="12" cy="8" r="4" /><path d="M4 21a8 8 0 0 1 16 0" /></>,
    heart: <path d="M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.7l-1.1-1.1a5.5 5.5 0 0 0-7.8 7.8l1.1 1.1L12 21l7.8-7.5 1.1-1.1a5.5 5.5 0 0 0-.1-7.8Z" />,
    cart: <><path d="M3 3h2l2.4 11.2a2 2 0 0 0 2 1.6h7.8a2 2 0 0 0 2-1.6L21 7H6" /><circle cx="10" cy="20" r="1" /><circle cx="18" cy="20" r="1" /></>,
    menu: <path d="M4 7h16M4 12h16M4 17h16" />, spark: <path d="m12 2 1.4 5.2L18 10l-4.6 2.8L12 18l-1.4-5.2L6 10l4.6-2.8L12 2ZM19 17l.6 2.1L22 20l-2.4.9L19 23l-.6-2.1L16 20l2.4-.9L19 17Z" />,
    arrow: <><path d="M5 12h14M14 7l5 5-5 5" /></>, check: <path d="m5 12 4 4L19 6" />, shield: <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Zm-3-10 2 2 4-4" />,
    truck: <><path d="M3 5h11v11H3zM14 9h4l3 3v4h-7z" /><circle cx="7" cy="18" r="2" /><circle cx="18" cy="18" r="2" /></>, box: <><path d="m4 7 8-4 8 4-8 4-8-4Z" /><path d="M4 7v10l8 4 8-4V7M12 11v10" /></>,
    refresh: <><path d="M20 7h-6V1" /><path d="M20 7a9 9 0 1 0 1 8" /></>, close: <path d="m6 6 12 12M18 6 6 18" />, plus: <path d="M12 5v14M5 12h14" />, send: <><path d="m22 2-7 20-4-9-9-4Z" /><path d="M22 2 11 13" /></>, paw: <><ellipse cx="7" cy="7" rx="2.2" ry="3"/><ellipse cx="12" cy="5.5" rx="2.2" ry="3"/><ellipse cx="17" cy="7" rx="2.2" ry="3"/><ellipse cx="5" cy="13" rx="2" ry="2.7"/><path d="M12 11c-3.1 0-6 3.4-6 6.1 0 2 2 2.9 6 2.9s6-.9 6-2.9c0-2.7-2.9-6.1-6-6.1Z"/></>,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}

function ProductVisual({ product, index = 0 }: { product: StoreProduct; index?: number }) {
  const [imageFailed, setImageFailed] = useState(false);
  const tones = ["sage", "peach", "mint", "lilac"];
  const code = (product.brand || product.title || "P").slice(0, 2).toUpperCase();
  return <div className={`product-visual visual-${tones[index % tones.length]}`}>
    {product.image_url && !imageFailed ? <img className="h-full w-full object-cover" src={product.image_url} alt={product.title} onError={() => setImageFailed(true)} /> : <>
      <span className="visual-orbit" /><div className="product-bag"><span className="bag-brand">PETMIND</span><span className="bag-code">{code}</span><span className="bag-leaf" /><span className="bag-line" /><span className="bag-weight">NATURAL · COMPLETE</span></div>
      <span className="kibble k1" /><span className="kibble k2" /><span className="kibble k3" />
    </>}
  </div>;
}

function ProductCard({ product, index = 0, onAdd, onDismiss, footer }: { product: StoreProduct; index?: number; onAdd: (product: StoreProduct) => void; onDismiss?: (product: StoreProduct) => void; footer?: React.ReactNode }) {
  const [liked, setLiked] = useState(false);
  const id = productId(product);
  const match = product.match_percent ?? 0;
  const safe = product.safe ?? (product.safe_badge === undefined ? null : product.safe_badge);
  return <article className="product-card"><div className="relative">
    <Link to={`/products/${encodeURIComponent(id)}`}><ProductVisual product={product} index={index} /></Link>
    <span className={`product-badge ${safe === false ? "product-badge-danger" : ""}`}>{safe === true ? "Без алергенів" : safe === false ? "Небезпечно" : "PetMind"}</span>
    <button className={`heart-button ${liked ? "liked" : ""}`} onClick={() => setLiked(!liked)} aria-label="Додати до обраного"><Icon name="heart" size={17} /></button>
    {match > 0 && <span className="match-badge" title="Показник збігу профілю, складу й історії покупок; це не ймовірність і не медична оцінка."><Icon name="spark" size={11} /> {match}% збігу</span>}
  </div><div className="px-1 pt-4"><div className="flex items-center justify-between"><span className="text-[10px] font-bold uppercase tracking-[0.12em] text-[#849085]">{product.species === "cat" ? "Котам" : product.species === "dog" ? "Собакам" : product.category || "Тваринам"}</span>
    {safe !== null && safe !== undefined && <span className={`text-xs font-bold ${safe ? "text-[#566259]" : "text-[#b33e2f]"}`}>{safe ? "Перевірено" : product.danger_reason || "Є ризик алергену"}</span>}</div>
    <Link className="mt-2 block text-[15px] font-extrabold tracking-[-0.02em]" to={`/products/${encodeURIComponent(id)}`}>{product.title}</Link>
    <p className="mt-1 text-xs text-[#7d867f]">{product.brand || product.category || "Товар для улюбленця"}</p>
    <div className="mt-4 flex items-end justify-between"><span className="text-lg font-extrabold">{product.price == null ? "Ціну уточнюйте" : `${product.price.toLocaleString("uk-UA")} ₴`}</span><button className="add-button" onClick={() => onAdd(product)} aria-label="Додати до кошика"><Icon name="plus" size={18} /></button></div>
    {safe === false && <p className="product-danger-copy mt-2">Попередження для обраної тварини: {product.danger_reason || "склад може містити її алерген"}. У кошику можна призначити товар іншій тварині.</p>}
    {footer}
    {onDismiss && <button className="mt-3 text-xs text-[#7d867f] underline" onClick={() => onDismiss(product)}>Не цікаво</button>}
  </div></article>;
}

function App() {
  const [search, setSearch] = useState("");
  const [authUser, setAuthUser] = useState<User | null>(readSavedAuthUser);
  const [userId, setUserId] = useState(() => readSavedAuthUser()?.id || "");
  const [selectedPetId, setSelectedPetId] = useState(() => localStorage.getItem("petly_pet_id") || "");
  const [cart, setCart] = useState<CartLine[]>([]);
  const token = localStorage.getItem("petmind_access_token");
  const usersState = useApiResource<User>(token ? "/auth/me" : null, token || "");
  const petsState = useApiResource<Pet[]>(userId ? `/users/${encodeURIComponent(userId)}/pets` : null, userId);
  const pets = petsState.loading ? [] : petsState.data || [];
  const selectedUser = usersState.data || (authUser?.id === userId ? authUser : undefined);
  const selectedPet = pets.find((pet) => pet.id === selectedPetId) || pets[0];
  useEffect(() => { if (usersState.data) { setAuthUser(usersState.data); setUserId(usersState.data.id); } }, [usersState.data]);
  useEffect(() => {
    if (!petsState.loading && pets.length && !pets.some((pet) => pet.id === selectedPetId)) setSelectedPetId(pets[0].id);
  }, [pets, petsState.loading, selectedPetId]);
  useEffect(() => { if (selectedPetId) localStorage.setItem("petly_pet_id", selectedPetId); else localStorage.removeItem("petly_pet_id"); }, [selectedPetId]);
  const addToCart = (product: StoreProduct) => {
    setCart((items) => [...items, { product, pet_id: selectedPet?.id || "" }]);
    if (userId && productId(product)) void apiRequest("/events", { method: "POST", body: JSON.stringify({ user_id: userId, pet_id: selectedPet?.id || null, product_id: productId(product), type: "add_to_cart" }) }).catch(() => undefined);
  };
  return <div className="min-h-screen bg-[#fbfaf6] text-[#213229]">
    <div className="announcement"><span>Безкоштовна доставка від 1 500 ₴</span><span className="hidden sm:inline">З турботою про кожного хвостика</span><span>0 800 33 14 14</span></div>
    <header className="site-header"><div className="container header-main"><button className="icon-button lg:hidden" aria-label="Меню"><Icon name="menu" /></button>
      <Link className="logo" to="/"><span className="logo-mark"><Icon name="paw" size={22} /></span><span>PetMind<span className="text-[#e26f45]">.</span></span></Link>
      <div className="search-shell"><Icon name="search" size={18} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Знайти корм, бренд або категорію..." aria-label="Пошук товарів" /><button onClick={() => setSearch("")} aria-label="Очистити пошук"><Icon name="close" size={15} /></button></div>
      <div className="header-actions">{authUser ? <><Link className="secondary-button" to="/account">{authUser.display_name || "Мій кабінет"}</Link><button className="secondary-button" onClick={() => { localStorage.removeItem("petmind_access_token"); localStorage.removeItem("petmind_auth_user"); setAuthUser(null); setUserId(""); setSelectedPetId(""); setCart([]); window.location.assign("/"); }}>Вийти</button></> : <Link className="secondary-button" to="/auth">Вхід / реєстрація</Link>}<Link className="cart-button" to="/cart"><Icon name="cart" /><span className="hidden sm:inline">Кошик</span>{cart.length > 0 && <b>{cart.length}</b>}</Link></div>
    </div><div className="nav-row"><nav className="container"><NavLink to="/" end>Головна</NavLink><NavLink to="/catalog">Каталог</NavLink><NavLink to="/catalog?species=dog">Собакам</NavLink><NavLink to="/catalog?species=cat">Котам</NavLink><NavLink to="/catalog?species=rabbit">Кролям</NavLink><Link to="/pharmacy">Ветаптека</Link><NavLink to="/brands">Бренди</NavLink><NavLink to="/recommendations"><Icon name="spark" size={13} /> AI-рекомендації</NavLink><Link className="ml-auto hidden xl:block" to="/catalog?sale=true">Акції</Link><NavLink to="/account">Кабінет</NavLink>{selectedUser?.is_admin && <NavLink to="/admin">Адмін</NavLink>}</nav></div></header>
    {usersState.loading && <div className="container py-3"><SkeletonLine /></div>}
    {usersState.error && <ErrorNotice message={usersState.error} />}
    {petsState.error && <ErrorNotice message={petsState.error} />}
    <Routes>
      <Route path="/" element={<HomePage userId={userId} pet={selectedPet} pets={pets} setSelectedPetId={setSelectedPetId} addToCart={addToCart} search={search} />} />
      <Route path="/catalog" element={<CatalogPage addToCart={addToCart} search={search} petId={selectedPet?.id} />} />
      <Route path="/products/:id" element={<ProductPage pet={selectedPet} addToCart={addToCart} />} />
      <Route path="/recommendations" element={userId ? <RecommendationsPage userId={userId} pet={selectedPet} addToCart={addToCart} /> : <Navigate to="/auth" replace />} />
      <Route path="/auth" element={<AuthPage onAuth={(user, token) => { localStorage.setItem("petmind_access_token", token); localStorage.setItem("petmind_auth_user", JSON.stringify(user)); setAuthUser(user); setUserId(user.id); setSelectedPetId(""); usersState.reload(); }} />} />
      <Route path="/admin" element={authUser?.is_admin && authUser.id === userId ? <AdminCatalogPage /> : <NotFound />} />
      <Route path="/account" element={userId ? <AccountPage userId={userId} pets={pets} selectedPetId={selectedPet?.id || ""} setSelectedPetId={setSelectedPetId} refreshPets={petsState.reload} /> : <Navigate to="/auth" replace />} />
      <Route path="/cart" element={<CartPage items={cart} setItems={setCart} userId={userId} pets={pets} />} />
      <Route path="/brands" element={<BrandsPage addToCart={addToCart} />} />
      <Route path="/pharmacy" element={<CatalogPage addToCart={addToCart} search={search} petId={selectedPet?.id} initialCategory="Ветаптека" />} />
      <Route path="/about" element={<InfoPage title="Про PetMind" text="PetMind — навчальний зоомаркет із каталогом товарів, профілями улюбленців і детермінованою перевіркою складу. Інформація про товари у демонстраційному каталозі може бути прикладом." />} />
      <Route path="/delivery" element={<InfoPage title="Доставка й оплата" text="Це демонстраційний проєкт: реальна оплата та доставка не виконуються. Замовлення зберігаються у локальній базі даних для перевірки функцій застосунку." />} />
      <Route path="/contacts" element={<InfoPage title="Контакти" text="PetMind — демонстраційний проєкт. Реальні телефон, адреса та служба підтримки не налаштовані. Не надсилайте сюди платіжні чи медичні дані." />} />
      <Route path="*" element={<NotFound />} />
    </Routes>
    <ChatWidget userId={userId} petId={selectedPet?.id || ""} addToCart={addToCart} />
    <Footer />
  </div>;
}

function HomePage({ userId, pet, pets, setSelectedPetId, addToCart, search }: { userId: string; pet?: Pet; pets: Pet[]; setSelectedPetId: (id: string) => void; addToCart: (p: StoreProduct) => void; search: string }) {
  const productParams = new URLSearchParams({ page: "1", sort: "relevance" });
  if (search) productParams.set("q", search);
  if (pet?.id) productParams.set("pet_id", pet.id);
  const productsState = useApiResource<ProductPage>(`/products?${productParams}`, search);
  const recommendationsPath = userId && pet?.id ? `/recommendations?user_id=${encodeURIComponent(userId)}&pet_id=${encodeURIComponent(pet.id)}&k=10&use_llm=true` : null;
  const recommendationsState = useApiResource<RecommendationItem[]>(recommendationsPath, `${userId}:${pet?.id || ""}`);
  const statsPath = userId && pet?.id ? `/stats?user_id=${encodeURIComponent(userId)}&pet_id=${encodeURIComponent(pet.id)}` : "/stats";
  const statsState = useApiResource<Stats>(statsPath, `${userId}:${pet?.id || ""}`);
  const products = productsState.data?.items || [];
  const featured = recommendationsState.data?.[0];
  return <main><section className="container hero-section"><div className="hero-content"><span className="hero-label"><Icon name="spark" size={14} /> Розумна турбота щодня</span><h1>Усе найкраще<br />для тих, хто <em>поруч</em></h1><p>Товари, що справді підходять вашому улюбленцю. Обирайте з турботою про його потреби.</p><div className="hero-buttons"><Link className="primary-button" to="/catalog">До каталогу <Icon name="arrow" size={17} /></Link><Link className="secondary-button" to="/recommendations"><Icon name="spark" size={16} /> Підібрати з AI</Link></div><div className="hero-proof"><div className="proof-faces"><span>П</span><span>М</span><span>Р</span></div><div><b>Турбота про улюбленців</b><small>Персональний підбір і перевірка складу</small></div></div></div><div className="hero-image"><img src="https://images.unsplash.com/photo-1450778869180-41d0601e046e?crop=entropy&cs=tinysrgb&fit=crop&fm=jpg&q=90&w=1400" alt="Собака та кіт разом" /><div className="hero-float-card"><span className="float-icon"><Icon name="shield" size={18} /></span><div><b>Перевірка складу</b><small>З урахуванням профілю тварини</small></div></div><div className="hero-decor">pet care</div></div></section>
    <section className="benefits"><div className="container benefits-grid"><div><span><Icon name="truck" /></span><p><b>Швидка доставка</b><small>1–2 дні по Україні</small></p></div><div><span><Icon name="shield" /></span><p><b>Безпечна оплата</b><small>Захищені платежі</small></p></div><div><span><Icon name="refresh" /></span><p><b>Легке повернення</b><small>Протягом 14 днів</small></p></div><div><span><Icon name="box" /></span><p><b>Лише оригінал</b><small>Офіційні постачальники</small></p></div></div></section>
    <section className="container section-space"><div className="section-heading"><div><span className="section-overline">Обирайте з любов’ю</span><h2>Для кожного улюбленця</h2></div><Link className="text-link" to="/catalog">Усі категорії <Icon name="arrow" size={15} /></Link></div><div className="category-grid">{[{ title: "Для собак", text: "Корм, ласощі та догляд", image: "https://images.unsplash.com/photo-1591946559594-8c6d3b7391eb?crop=entropy&cs=tinysrgb&fit=crop&fm=jpg&q=85&w=700", species: "dog" }, { title: "Для котів", text: "Усе для муркотливих", image: "https://images.unsplash.com/photo-1490650034439-fd184c3c86a5?crop=entropy&cs=tinysrgb&fit=crop&fm=jpg&q=85&w=700", species: "cat" }, { title: "Ветаптека", text: "Вітаміни та захист", image: "https://images.unsplash.com/photo-1714068691210-073dc52c6c1d?crop=entropy&cs=tinysrgb&fit=crop&fm=jpg&q=85&w=700", species: "" }].map((category, index) => <Link className={`category-card cat-${index + 1}`} key={category.title} to={category.species ? `/catalog?species=${category.species}` : "/catalog?category=Ветаптека"}><img src={category.image} alt={category.title} /><span className="category-overlay" /><span className="category-copy"><b>{category.title}</b><small>{category.text}</small><i><Icon name="arrow" size={15} /></i></span></Link>)}</div></section>
    <section className="container section-space"><div className="section-heading"><div><span className="section-overline">Популярне зараз</span><h2>Рекомендовані товари</h2></div><Link className="text-link" to="/catalog">Увесь каталог <Icon name="arrow" size={15} /></Link></div>{productsState.loading ? <SkeletonGrid /> : productsState.error ? <ErrorNotice message={productsState.error} /> : <ProductGrid products={products.slice(0, 4)} addToCart={addToCart} />}</section>
    <section className="container"><div className="ai-banner"><div className="ai-copy"><span className="ai-kicker"><Icon name="spark" size={14} /> PetMind AI</span><h2>Персональна добірка<br />для {pet?.name || "вашого улюбленця"}</h2><p>{statsState.loading ? "Завантажуємо дані профілю та каталогу…" : statsState.error ? statsState.error : `Проаналізовано ${statsState.data?.characteristics_count ?? 0} характеристик і склад ${statsState.data?.composition_product_count ?? 0} товарів.`}</p><div className="pet-chip"><div><b>{pet ? `${pet.name}${pet.weight_kg ? ` · ${pet.weight_kg} кг` : ""}` : "Додайте профіль тварини"}</b><small>{pet?.breed || (pet?.species === "cat" ? "Кіт" : pet?.species === "dog" ? "Собака" : "Профіль потрібен для персоналізації")}</small></div><label className="pet-change">Змінити<select aria-label="Змінити обрану тварину" value={pet?.id || ""} onChange={(event) => setSelectedPetId(event.target.value)}><option value="" disabled>Оберіть</option>{pets.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label></div>{recommendationsState.error && <p role="status">{recommendationsState.error}</p>}</div><div className="ai-result">{recommendationsState.loading ? <div className="p-8"><SkeletonLine /><SkeletonLine /></div> : featured ? <><div className="ai-score"><span>{featured.match_percent}%</span><small>сумісність</small></div><ProductVisual product={featured} /><div className="ai-product-copy"><small>Найкращий вибір</small><b>{featured.title}</b>{featured.safe_badge && <span><Icon name="check" size={14} /> Без алергенів для {pet?.name || "тварини"}</span>}<Link to="/recommendations">Усі рекомендації</Link></div></> : <div className="ai-product-copy"><small>PetMind AI</small><b>{pet ? "Наразі немає рекомендацій" : "Оберіть профіль тварини"}</b><Link to="/recommendations">Переглянути</Link></div>}</div></div></section>
    <section className="story-section mt-20"><div className="container story-grid"><div className="story-image"><img src="https://images.unsplash.com/photo-1601758228006-964e41e5e8eb?crop=entropy&cs=tinysrgb&fit=crop&fm=jpg&q=85&w=1200" alt="Собака їсть корм" /><span>Якість, якій довіряють</span></div><div className="story-copy"><span className="section-overline">Наша філософія</span><h2>Не просто магазин.<br />Партнер у турботі.</h2><p>Ми поєднали експертизу, перевірені товари та сучасні технології, щоб допомогти обрати потрібне.</p><ul><li><Icon name="check" size={15} /> Перевіряємо склад товарів</li><li><Icon name="check" size={15} /> Враховуємо потреби тварини</li><li><Icon name="check" size={15} /> Пояснюємо рекомендації</li></ul></div></div></section>
  </main>;
}

function CatalogPage({ addToCart, search, petId, initialCategory }: { addToCart: (p: StoreProduct) => void; search: string; petId?: string; initialCategory?: string }) {
  const [params, setParams] = useSearchParams(); const [category, setCategory] = useState(params.get("species") || initialCategory || (params.get("sale") ? "sale" : params.get("brand") ? "brand" : "all")); const [page, setPage] = useState(1); const [brand, setBrand] = useState(params.get("brand") === "all" ? "" : params.get("brand") || "");
  useEffect(() => { setCategory(params.get("species") || (params.get("sale") ? "sale" : params.get("brand") ? "brand" : "all")); }, [params]);
  useEffect(() => { setBrand(params.get("brand") === "all" ? "" : params.get("brand") || ""); }, [params]);
  const query = new URLSearchParams({ page: String(page), sort: "relevance" }); if (search) query.set("q", search); if (!["all", "brand", "sale", "Ветаптека"].includes(category)) query.set("species", category); if (category === "brand" && brand) query.set("brand", brand); if (petId) query.set("pet_id", petId); if (category === "Ветаптека") query.set("category", "Ветаптека"); else if (params.get("category")) query.set("category", params.get("category")!);
  const productsState = useApiResource<ProductPage>(`/products?${query}`, `${category}:${brand}:${page}:${petId || ""}`);
  const products = productsState.data?.items || [];
  const total = productsState.data?.total || 0;
  const filters = [{ label: "Усі товари", value: "all" }, { label: "Собакам", value: "dog" }, { label: "Котам", value: "cat" }, { label: "Кролям", value: "rabbit" }, { label: "Птахам", value: "bird" }, { label: "Гризунам", value: "rodent" }, { label: "Рептиліям", value: "reptile" }, { label: "Рибам", value: "fish" }, { label: "Бренди", value: "brand" }, { label: "Акції", value: "sale" }];
  return <main className="container section-space"><div className="section-heading products-heading"><div><span className="section-overline">Каталог PetMind</span><h1>Товари для улюбленців</h1><p>{total ? `${total} товарів` : "Знайдіть потрібне для свого улюбленця"}</p></div><div className="filter-tabs">{filters.map((filter) => <button key={filter.value} className={category === filter.value ? "active" : ""} onClick={() => { setCategory(filter.value); setPage(1); if (filter.value === "dog" || filter.value === "cat") setParams({ species: filter.value }); else if (filter.value === "brand") setParams({ brand: "all" }); else if (filter.value === "sale") setParams({ sale: "true" }); else setParams({}); }}>{filter.label}</button>)}</div></div>{category === "brand" && <input className="form-control mb-6" aria-label="Фільтр бренду" placeholder="Введіть бренд" value={brand} onChange={(event) => setBrand(event.target.value)} />}{category === "sale" && <div className="ai-banner mb-8"><div className="ai-copy"><span className="ai-kicker">Акції PetMind</span><h2>Спеціальні пропозиції</h2><p>Акційні товари з’являться в каталозі, щойно будуть доступні.</p></div></div>}{productsState.loading ? <SkeletonGrid /> : productsState.error ? <ErrorNotice message={productsState.error} /> : category !== "sale" && <ProductGrid products={products} addToCart={addToCart} />}<div className="mt-8 flex justify-center gap-3"><button className="secondary-button" disabled={page <= 1} onClick={() => setPage(page - 1)}>Назад</button><button className="secondary-button" disabled={products.length === 0 || page * 24 >= total} onClick={() => setPage(page + 1)}>Далі</button></div></main>;
}

function ProductPage({ pet, addToCart }: { pet?: Pet; addToCart: (p: StoreProduct) => void }) {
  const { id = "" } = useParams();
  const productState = useApiResource<ProductDetail>(`/products/${encodeURIComponent(id)}${pet?.id ? `?pet_id=${encodeURIComponent(pet.id)}` : ""}`, pet?.id || "");
  const product = productState.data;
  if (productState.error) return <main className="container section-space"><ErrorNotice message={productState.error} /><Link className="primary-button" to="/catalog">До каталогу</Link></main>;
  if (productState.loading || !product) return <main className="container section-space"><SkeletonProduct /></main>;
  return <main className="container section-space"><div className="story-grid"><ProductVisual product={product} /><div className="story-copy"><span className="section-overline">{product.brand || product.category}</span><h1>{product.title}</h1><p>{product.description}</p><strong className="text-2xl">{product.price == null ? "Ціну уточнюйте" : `${product.price.toLocaleString("uk-UA")} ₴`}</strong><p className={product.safe === false ? "product-danger-copy" : ""}>{product.safe === false ? `Попередження для ${pet?.name || "обраної тварини"}: ${product.danger_reason || "склад містить можливий алерген"}. У кошику можна обрати іншу тварину.` : pet ? `Перевірено для ${pet.name}.` : "Для перевірки складу увійдіть та оберіть профіль тварини."}</p><button className="primary-button" onClick={() => addToCart(product)}>Додати до кошика</button><h2 className="mt-8">Склад</h2><p>{product.ingredients_text || "Склад не вказано"}</p><ul>{product.ingredients.map((ingredient) => <li className={ingredient.is_allergen ? "ingredient-highlight" : ""} key={`${ingredient.position}-${ingredient.name}`}>{ingredient.name}{ingredient.is_allergen ? " — можливий алерген" : ""}</li>)}</ul><p>Показники складу та поживності допомагають порівняти позиції; це не відсоток імовірності користі.</p></div></div></main>;
}

function RecommendationsPage({ userId, pet, addToCart }: { userId: string; pet?: Pet; addToCart: (p: StoreProduct) => void }) {
  const path = userId && pet?.id ? `/recommendations?user_id=${encodeURIComponent(userId)}&pet_id=${encodeURIComponent(pet.id)}&k=10&use_llm=true` : null;
  const state = useApiResource<RecommendationItem[]>(path, `${userId}:${pet?.id || ""}`);
  const rows = state.data || [];
  const [dismissed, setDismissed] = useState<string[]>([]);
  const dismiss = async (product: StoreProduct) => {
    if (!userId || !pet?.id) return;
    try { await apiRequest("/events", { method: "POST", body: JSON.stringify({ user_id: userId, pet_id: pet.id, product_id: productId(product), type: "dismiss" }) }); setDismissed((ids) => [...ids, productId(product)]); }
    catch (error) { window.alert(error instanceof Error ? error.message : "Не вдалося записати реакцію."); }
  };
  const visibleRows = rows.filter((row) => !dismissed.includes(row.item_id));
  return <main className="container research-page"><section className="research-hero"><span className="hero-label"><Icon name="spark" size={14} /> PetMind AI</span><h1>Рекомендації<br /><em>для {pet?.name || "вашого улюбленця"}</em></h1><p>Підбір враховує вподобання, профіль тварини та безпеку складу.</p><Link className="secondary-button" to="/account">Керувати профілями</Link></section>{state.loading ? <SkeletonGrid /> : state.error ? <ErrorNotice message={state.error} /> : visibleRows.length ? <><div className="section-heading"><div><span className="section-overline">Перевірені товари</span><h2>Ваші рекомендації</h2></div></div>{rows.some((item) => !item.llm_used) && <div className="notice-banner">LLM недоступна — показано базові рекомендації.</div>}<div className="products-grid">{visibleRows.slice(0, 10).map((row, index) => <RecommendationCard key={row.item_id} item={row} petId={pet?.id || ""} index={index} addToCart={addToCart} onDismiss={dismiss} />)}</div></> : <div className="empty-state"><Icon name="spark" size={28} /><h3>{pet ? "Поки немає рекомендацій" : "Додайте профіль тварини"}</h3><p>Профіль допоможе врахувати потреби та алергії.</p><Link className="primary-button" to="/account">До профілю</Link></div>}</main>;
}

function AccountPage({ userId, pets, selectedPetId, setSelectedPetId, refreshPets }: { userId: string; pets: Pet[]; selectedPetId: string; setSelectedPetId: (id: string) => void; refreshPets: () => void }) {
  const remindersState = useApiResource<ReminderItem[]>(userId && selectedPetId ? `/reminders?user_id=${encodeURIComponent(userId)}&pet_id=${encodeURIComponent(selectedPetId)}` : null, selectedPetId);
  const selectedPet = pets.find((pet) => pet.id === selectedPetId);
  const [isNew, setIsNew] = useState(false); const [name, setName] = useState(""); const [species, setSpecies] = useState<string>("cat");
  const [birthDate, setBirthDate] = useState(""); const [weight, setWeight] = useState(""); const [breed, setBreed] = useState(""); const [allergens, setAllergens] = useState<string[]>([]);
  const [restrictions, setRestrictions] = useState(""); const [lifestyle, setLifestyle] = useState(""); const [saving, setSaving] = useState(false); const [formError, setFormError] = useState(""); const [notice, setNotice] = useState("");
  useEffect(() => {
    if (isNew) return;
    setName(selectedPet?.name || ""); setSpecies(selectedPet?.species || "cat"); setBirthDate(selectedPet?.birth_date ? formatDateUk(selectedPet.birth_date) : "");
    setWeight(selectedPet?.weight_kg == null ? "" : String(selectedPet.weight_kg)); setBreed(selectedPet?.breed || "");
    setAllergens(selectedPet?.allergens || []); setRestrictions((selectedPet?.diet_restrictions || []).join(", ")); setLifestyle(selectedPet?.lifestyle || "");
  }, [selectedPetId, isNew, selectedPet]);
  async function savePet(event: React.FormEvent) {
    event.preventDefault(); if (!userId) { setFormError("Оберіть користувача для збереження профілю."); return; }
    setSaving(true); setFormError(""); setNotice("");
    const isoBirthDate = parseDateUk(birthDate);
    if (birthDate && !isoBirthDate) { setSaving(false); setFormError("Введіть дату у форматі ДД.ММ.РРРР, наприклад 14.05.2020."); return; }
    const payload: PetInput = { name, species, birth_date: isoBirthDate, breed: breed || null, weight_kg: weight ? Number(weight) : null,
      allergens, diet_restrictions: restrictions.split(",").map((value) => value.trim()).filter(Boolean), lifestyle: lifestyle || null, photo_url: isNew ? null : selectedPet?.photo_url || null };
    try {
      const result = isNew
        ? await apiRequest<Pet>(`/users/${encodeURIComponent(userId)}/pets`, { method: "POST", body: JSON.stringify(payload) })
        : await apiRequest<Pet>(`/pets/${encodeURIComponent(selectedPetId)}`, { method: "PUT", body: JSON.stringify(payload) });
      refreshPets(); setSelectedPetId(result.id); setIsNew(false); setNotice("Профіль збережено.");
    } catch (error) { setFormError(error instanceof Error ? error.message : "Не вдалося зберегти профіль."); }
    finally { setSaving(false); }
  }
  async function reorder(reminder: ReminderItem) {
    try { await apiRequest("/purchases", { method: "POST", body: JSON.stringify({ user_id: userId, pet_id: selectedPetId, items: [{ product_id: reminder.item_id }] }) }); setNotice("Повторне замовлення оформлено."); remindersState.reload(); }
    catch (error) { setFormError(error instanceof Error ? error.message : "Не вдалося оформити замовлення."); }
  }
  const allergenOptions = ["птиця", "курятина", "яловичина", "риба", "пшениця", "злаки з глютеном", "молочні продукти"];
  return <main className="container section-space"><div className="section-heading"><div><span className="section-overline">Особистий кабінет</span><h1>Профілі та нагадування</h1></div></div>
    <section className="research-panel"><div className="section-heading"><div><h2>Профілі тварин</h2></div><button className="secondary-button" onClick={() => { setIsNew(true); setName(""); setSpecies("cat"); setBirthDate(""); setWeight(""); setBreed(""); setAllergens([]); setRestrictions(""); setLifestyle(""); }}>Додати тварину</button></div>
      <label className="field-label">Обрана тварина<select className="form-control" value={selectedPetId} onChange={(event) => { setIsNew(false); setSelectedPetId(event.target.value); }}>{pets.map((pet) => <option key={pet.id} value={pet.id}>{pet.name} · {pet.species === "cat" ? "кіт" : "собака"}</option>)}</select></label>
      <form className="pet-form" onSubmit={savePet}><label className="field-label">Ім’я<input className="form-control" required maxLength={160} value={name} onChange={(e) => setName(e.target.value)} /></label>
        <label className="field-label">Вид<select className="form-control" value={species} onChange={(e) => setSpecies(e.target.value)}><option value="cat">Кіт</option><option value="dog">Собака</option><option value="rabbit">Кролик</option><option value="bird">Птах</option><option value="rodent">Гризун</option><option value="reptile">Рептилія</option><option value="fish">Риба</option><option value="other">Інша тварина</option></select></label>
        <label className="field-label">Дата народження<input className="form-control" type="text" inputMode="numeric" placeholder="ДД.ММ.РРРР" maxLength={10} value={birthDate} onChange={(e) => setBirthDate(e.target.value.replace(/[^0-9.]/g, "").slice(0, 10))} /></label>
        <label className="field-label">Маса, кг<input className="form-control" type="number" min="0.1" step="0.1" value={weight} onChange={(e) => setWeight(e.target.value)} /></label>
        <label className="field-label">Порода<input className="form-control" value={breed} onChange={(e) => setBreed(e.target.value)} /></label>
        <label className="field-label">Спосіб життя<select className="form-control" value={lifestyle} onChange={(e) => setLifestyle(e.target.value)}><option value="">Не вказано</option><option value="активний">Активний</option><option value="помірний">Помірний</option><option value="домашній">Домашній</option><option value="стерилізований">Стерилізований</option></select></label>
        <fieldset className="field-set"><legend>Алергени</legend>{allergenOptions.map((allergen) => <label key={allergen}><input type="checkbox" checked={allergens.includes(allergen)} onChange={(event) => setAllergens(event.target.checked ? [...allergens, allergen] : allergens.filter((value) => value !== allergen))} /> {allergen}</label>)}</fieldset>
        <label className="field-label md:col-span-2">Дієтичні обмеження (через кому)<input className="form-control" value={restrictions} onChange={(e) => setRestrictions(e.target.value)} /></label>
        <div className="md:col-span-2"><button className="primary-button" disabled={saving || (!isNew && !selectedPetId)}>{saving ? "Збереження…" : isNew ? "Створити профіль" : "Зберегти зміни"}</button>{formError && <p className="form-error" role="alert">{formError}</p>}{notice && <p className="form-notice" role="status">{notice}</p>}</div>
      </form>
    </section>
    <section className="evaluation-section"><div className="section-heading"><div><span className="section-overline">Для обраної тварини</span><h2>Нагадування</h2></div></div>
      {remindersState.loading ? <SkeletonLine /> : remindersState.error ? <ErrorNotice message={remindersState.error} /> : remindersState.data?.length ? remindersState.data.map((reminder) => <article className="reminder-row" key={`${reminder.item_id}-${reminder.expected_date}`}><div className="mini-bag">{reminder.days_left}</div><div><b>{reminder.title}</b><small>{reminder.text}</small><small>{reminder.expected_date} · {reminder.severity}</small></div><button className="secondary-button" onClick={() => void reorder(reminder)}>Замовити знову</button></article>) : <p>Наразі нагадувань немає.</p>}
    </section></main>;
}

function CartPage({ items, setItems, userId, pets }: { items: CartLine[]; setItems: (items: CartLine[]) => void; userId: string; pets: Pet[] }) {
  const [message, setMessage] = useState("");
  const total = items.reduce((sum, line) => sum + (line.product.price || 0), 0);
  async function checkout() {
    if (!userId) { setMessage("Увійдіть в обліковий запис, щоб оформити замовлення."); return; }
    if (!items.every((line) => line.pet_id)) { setMessage("Для кожного товару оберіть тварину."); return; }
    try {
      await apiRequest("/purchases", { method: "POST", body: JSON.stringify({ user_id: userId, items: items.map((line) => ({ product_id: productId(line.product), pet_id: line.pet_id })) }) });
      setItems([]); setMessage("Замовлення оформлено.");
    } catch (error) { setMessage(error instanceof Error ? error.message : "Не вдалося оформити замовлення."); }
  }
  function updatePet(index: number, pet_id: string) { setItems(items.map((line, i) => i === index ? { ...line, pet_id } : line)); }
  return <main className="container section-space"><div className="section-heading"><div><span className="section-overline">Ваше замовлення</span><h1>Кошик</h1></div></div>{items.length ? <section className="research-panel">{items.map((line, index) => <CartLine key={`${productId(line.product)}-${index}`} line={line} pets={pets} onPetChange={(petId) => updatePet(index, petId)} onRemove={() => setItems(items.filter((_, i) => i !== index))} />)}<div className="cart-total"><span>Разом</span><b>{total.toLocaleString("uk-UA")} ₴</b></div><button className="primary-button" onClick={checkout}>Оформити замовлення</button>{message && <p className="form-error" role="alert">{message}</p>}</section> : <div className="empty-state"><span><Icon name="cart" size={30} /></span><h3>Кошик поки порожній</h3><p>Додайте товари, які сподобались.</p><Link className="primary-button" to="/catalog">До покупок</Link></div>}</main>;
}

function CartLine({ line, pets, onPetChange, onRemove }: { line: CartLine; pets: Pet[]; onPetChange: (id: string) => void; onRemove: () => void }) {
  const id = productId(line.product);
  const safety = useApiResource<ProductDetail>(line.pet_id ? `/products/${encodeURIComponent(id)}?pet_id=${encodeURIComponent(line.pet_id)}` : null, line.pet_id);
  const product = safety.data || null;
  return <article className="cart-line mb-5"><div className="mini-bag">{(line.product.brand || "P").slice(0, 2)}</div><div className="min-w-0 flex-1"><b>{line.product.title}</b><small>{line.product.brand || line.product.category}</small><label className="field-label mt-2">Для кого<select className="form-control" value={line.pet_id} onChange={(event) => onPetChange(event.target.value)}><option value="">Оберіть тварину</option>{pets.map((pet) => <option key={pet.id} value={pet.id}>{pet.name} · {pet.species}</option>)}</select></label>{line.pet_id && safety.loading && <small>Перевіряємо склад…</small>}{product?.safe === false && <p className="product-danger-copy" role="alert">Не підходить для {pets.find((pet) => pet.id === line.pet_id)?.name}: {product.danger_reason || "виявлено можливий алерген"}. Оберіть іншу тварину, якщо для неї склад безпечний.</p>}</div><strong>{line.product.price?.toLocaleString("uk-UA")} ₴</strong><button className="icon-button" onClick={onRemove} aria-label="Вилучити"><Icon name="close" /></button></article>;
}

function RecommendationCard({ item, petId, index, addToCart, onDismiss }: { item: RecommendationItem; petId: string; index: number; addToCart: (p: StoreProduct) => void; onDismiss: (p: StoreProduct) => void }) {
  const detailState = useApiResource<ProductDetail>(petId ? `/products/${encodeURIComponent(item.item_id)}?pet_id=${encodeURIComponent(petId)}` : null, petId);
  const product: StoreProduct = { ...item, id: item.item_id, safe_badge: item.safe_badge };
  const ingredients = detailState.data?.ingredients;
  return <ProductCard product={product} index={index} onAdd={addToCart} onDismiss={onDismiss} footer={<div className="recommendation-extra"><p>{item.reason}</p><div className="ingredient-list">{ingredients?.length ? ingredients.slice(0, 5).map((ingredient) => <span className={ingredient.is_allergen ? "ingredient-highlight" : "ingredient-chip"} key={`${ingredient.position}-${ingredient.name}`}>{ingredient.name}</span>) : item.composition_summary.map((ingredient, ingredientIndex) => <span className="ingredient-chip" key={`${ingredient}-${ingredientIndex}`}>{ingredient}</span>)}</div></div>} />;
}

function ChatWidget({ userId, petId, addToCart }: { userId: string; petId: string; addToCart: (p: StoreProduct) => void }) {
  const [open, setOpen] = useState(false); const [messages, setMessages] = useState<ChatMessage[]>([]); const [text, setText] = useState(""); const [products, setProducts] = useState<RecommendationItem[]>([]); const [busy, setBusy] = useState(false); const [error, setError] = useState("");
  async function send(event: React.FormEvent) {
    event.preventDefault(); const content = text.trim(); if (!content || !userId || !petId || busy) return;
    const nextMessages: ChatMessage[] = [...messages.slice(-28), { role: "user", content }]; setMessages(nextMessages); setText(""); setBusy(true); setError("");
    try { const result = await apiRequest<ChatResponse>("/chat", { method: "POST", body: JSON.stringify({ user_id: userId, pet_id: petId, messages: nextMessages }) }); setMessages([...nextMessages, { role: "assistant", content: result.reply }]); setProducts(result.products); }
    catch (requestError) { setError(requestError instanceof Error ? requestError.message : "Не вдалося надіслати повідомлення."); }
    finally { setBusy(false); }
  }
  return <div className="chat-widget"><button className="chat-toggle primary-button" onClick={() => setOpen((value) => !value)}><Icon name="spark" size={17} /> Спитати PetMind AI</button>{open && <section className="chat-panel" aria-label="Чат PetMind AI"><header><b>PetMind AI</b><button className="icon-button" onClick={() => setOpen(false)} aria-label="Закрити чат"><Icon name="close" /></button></header><div className="chat-messages">{messages.length === 0 && <p>Запитайте про товари, склад або рекомендації.</p>}{messages.map((message, index) => <p className={`chat-bubble ${message.role}`} key={`${message.role}-${index}`}>{message.content}</p>)}{busy && <SkeletonLine />}{error && <ErrorNotice message={error} />}{products.length > 0 && <div className="chat-products">{products.map((item, index) => <ProductCard key={item.item_id} product={{ ...item, id: item.item_id }} index={index} onAdd={addToCart} />)}</div>}</div><form onSubmit={send}><input value={text} onChange={(event) => setText(event.target.value)} placeholder={petId ? "Ваше запитання…" : "Спочатку оберіть тварину"} disabled={!petId || busy} /><button className="add-button" aria-label="Надіслати" disabled={!petId || busy || !text.trim()}><Icon name="send" size={17} /></button></form></section>}</div>;
}

function SkeletonLine() { return <div className="skeleton-line" aria-label="Завантаження" />; }
function SkeletonGrid() { return <div className="products-grid">{Array.from({ length: 4 }, (_, index) => <div className="product-card skeleton-card" key={index}><div className="skeleton-image" /><SkeletonLine /><SkeletonLine /></div>)}</div>; }
function SkeletonProduct() { return <div className="story-grid"><div className="skeleton-image" /><div><SkeletonLine /><SkeletonLine /><SkeletonLine /></div></div>; }
function ErrorNotice({ message }: { message: string }) { return <div className="api-error" role="alert">{message}</div>; }

function ProductGrid({ products, addToCart }: { products: StoreProduct[]; addToCart: (p: StoreProduct) => void }) { return products.length ? <div className="products-grid">{products.map((product, index) => <ProductCard key={productId(product)} product={product} index={index} onAdd={addToCart} />)}</div> : <div className="empty-state"><Icon name="search" size={28} /><h3>Товарів поки немає</h3><p>Спробуйте змінити фільтри або перевірте, чи запущений API.</p></div>; }
function NotFound() { return <main className="container section-space"><div className="empty-state"><h1>Сторінку не знайдено</h1><Link className="primary-button" to="/">На головну</Link></div></main>; }
function AdminCatalogPage() {
  const productsState = useApiResource<ProductPage>("/products?page=1&sort=title");
  const [selectedId, setSelectedId] = useState("");
  const [isNew, setIsNew] = useState(true);
  const selectedProductState = useApiResource<ProductDetail>(!isNew && selectedId ? `/products/${encodeURIComponent(selectedId)}` : null, selectedId);
  const selected = selectedProductState.data;
  const [id, setId] = useState(""); const [title, setTitle] = useState(""); const [brand, setBrand] = useState(""); const [category, setCategory] = useState("");
  const [species, setSpecies] = useState<"cat" | "dog">("cat"); const [price, setPrice] = useState("");
  const [imageUrl, setImageUrl] = useState(""); const [description, setDescription] = useState(""); const [ingredients, setIngredients] = useState("");
  const [message, setMessage] = useState(""); const [saving, setSaving] = useState(false);
  useEffect(() => {
    if (!selected || isNew) return;
    setId(selected.id); setTitle(selected.title); setBrand(selected.brand || ""); setCategory(selected.category || ""); setSpecies(selected.species === "dog" ? "dog" : "cat");
    setPrice(selected.price == null ? "" : String(selected.price)); setImageUrl(selected.image_url || "");
    setDescription(selected.description || ""); setIngredients(selected.ingredients_text || "");
  }, [selected, isNew]);
  function startNew() {
    setIsNew(true); setSelectedId(""); setId(""); setTitle(""); setBrand(""); setCategory(""); setSpecies("cat");
    setPrice(""); setImageUrl(""); setDescription(""); setIngredients(""); setMessage("");
  }
  async function save(event: React.FormEvent) {
    event.preventDefault(); setSaving(true); setMessage("");
    const payload = { ...(isNew ? { id: id.trim() } : {}), title: title.trim(), brand: brand || null, category: category || null, species,
      price: price ? Number(price) : null, image_url: imageUrl || null, description: description || null,
      ingredients_text: ingredients || null };
    try {
      await apiRequest(isNew ? "/admin/products" : `/admin/products/${encodeURIComponent(id)}`, {
        method: isNew ? "POST" : "PUT", body: JSON.stringify(payload),
      });
      setMessage("Товар збережено в каталозі."); productsState.reload();
      if (isNew) startNew();
    } catch (error) { setMessage(error instanceof Error ? error.message : "Не вдалося зберегти товар."); }
    finally { setSaving(false); }
  }
  return <main className="container section-space"><div className="section-heading"><div><span className="section-overline">Профіль адміністратора</span><h1>Керування товарами</h1></div><button className="secondary-button" onClick={startNew}>Додати товар</button></div>
    <div className="research-panel"><label className="field-label">Товар для редагування<select className="form-control" value={isNew ? "" : selectedId} onChange={(event) => { setSelectedId(event.target.value); setIsNew(false); setMessage(""); }}><option value="">Новий товар…</option>{(productsState.data?.items || []).map((item) => <option key={item.id} value={item.id}>{item.title}</option>)}</select></label>
      {productsState.error && <ErrorNotice message={productsState.error} />}
      <form className="pet-form" onSubmit={save}>
        <label className="field-label">ID товару<input className="form-control" required disabled={!isNew} value={id} onChange={(event) => setId(event.target.value)} /></label>
        <label className="field-label">Назва<input className="form-control" required value={title} onChange={(event) => setTitle(event.target.value)} /></label>
        <label className="field-label">Бренд<input className="form-control" value={brand} onChange={(event) => setBrand(event.target.value)} /></label>
        <label className="field-label">Категорія<input className="form-control" value={category} onChange={(event) => setCategory(event.target.value)} /></label>
        <label className="field-label">Для кого<select className="form-control" value={species} onChange={(event) => setSpecies(event.target.value as "cat" | "dog")}><option value="cat">Коти</option><option value="dog">Собаки</option></select></label>
        <label className="field-label">Ціна, ₴<input className="form-control" type="number" min="0" step="0.01" value={price} onChange={(event) => setPrice(event.target.value)} /></label>
        <label className="field-label">Посилання на фото<input className="form-control" type="url" value={imageUrl} onChange={(event) => setImageUrl(event.target.value)} /></label>
        <label className="field-label">Опис<textarea className="form-control" rows={4} value={description} onChange={(event) => setDescription(event.target.value)} /></label>
        <label className="field-label">Склад<input className="form-control" value={ingredients} onChange={(event) => setIngredients(event.target.value)} /></label>
        {message && <p role="status">{message}</p>}<button className="primary-button" disabled={saving}>{saving ? "Зберігаємо…" : "Зберегти товар"}</button>
      </form>
    </div>
  </main>;
}

function AuthPage({ onAuth }: { onAuth: (user: User, token: string) => void }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState(""); const [password, setPassword] = useState(""); const [displayName, setDisplayName] = useState("");
  const [busy, setBusy] = useState(false); const [message, setMessage] = useState(""); const navigate = useNavigate();
  async function submit(event: React.FormEvent) {
    event.preventDefault(); setBusy(true); setMessage("");
    try {
      const path = mode === "register" ? "/auth/register" : "/auth/login";
      const body = mode === "register" ? { email, password, display_name: displayName } : { email, password };
      const result = await apiRequest<AuthResponse>(path, { method: "POST", body: JSON.stringify(body) });
      onAuth(result.user, result.access_token); navigate(result.user.is_admin ? "/admin" : "/account");
    } catch (error) { setMessage(error instanceof Error ? error.message : "Не вдалося виконати запит."); }
    finally { setBusy(false); }
  }
  return <main className="container section-space"><section className="research-panel max-w-xl mx-auto"><div className="section-heading"><div><span className="section-overline">PetMind</span><h1>{mode === "register" ? "Створити акаунт" : "Увійти"}</h1><p>Профілі улюбленців і замовлення збережуться у вашому акаунті.</p></div></div>
    <div className="flex gap-3 mb-6"><button className={mode === "register" ? "primary-button" : "secondary-button"} type="button" onClick={() => { setMode("register"); setMessage(""); }}>Реєстрація</button><button className={mode === "login" ? "primary-button" : "secondary-button"} type="button" onClick={() => { setMode("login"); setMessage(""); }}>Вхід</button></div>
    <form className="pet-form" onSubmit={submit}>{mode === "register" && <label className="field-label">Ваше ім’я<input className="form-control" required maxLength={200} autoComplete="name" value={displayName} onChange={(event) => setDisplayName(event.target.value)} /></label>}
      <label className="field-label">Електронна пошта<input className="form-control" required type="email" autoComplete="email" value={email} onChange={(event) => setEmail(event.target.value)} /></label>
      <label className="field-label">Пароль<input className="form-control" required type="password" minLength={mode === "register" ? 8 : 1} autoComplete={mode === "register" ? "new-password" : "current-password"} value={password} onChange={(event) => setPassword(event.target.value)} /></label>
      <button className="primary-button" disabled={busy}>{busy ? "Зачекайте…" : mode === "register" ? "Зареєструватися" : "Увійти"}</button>{message && <p className="form-error" role="alert">{message}</p>}
    </form>
  </section></main>;
}
function Footer() { return <footer><div className="container footer-main"><div><Link className="logo" to="/"><span className="logo-mark"><Icon name="paw" size={22} /></span><span>PetMind<span className="text-[#e26f45]">.</span></span></Link><p>Турбота, підкріплена знаннями<br />та сучасними технологіями.</p></div><div><b>Покупцям</b><Link to="/catalog">Каталог</Link><Link to="/delivery">Доставка й оплата</Link><Link to="/account">Особистий кабінет</Link></div><div><b>PetMind</b><Link to="/about">Про нас</Link><Link to="/recommendations">Рекомендації</Link><Link to="/contacts">Контакти</Link></div><div><b>Ветаптека</b><Link to="/pharmacy">Ветпрепарати</Link><Link to="/brands">Бренди</Link></div></div><div className="container footer-bottom"><span>© 2026 PetMind</span><span>Піклуємося про кожного улюбленця</span></div></footer>; }

function InfoPage({ title, text }: { title: string; text: string }) { return <main className="container section-space"><span className="section-overline">PetMind</span><h1>{title}</h1><section className="research-panel mt-6"><p>{text}</p></section></main>; }

function BrandsPage({ addToCart }: { addToCart: (p: StoreProduct) => void }) {
  const brands = [
    { name: "Royal Canin", image: "https://www.petcorner.rs/cdn/shop/files/royal-canin-sterilised-37-suva-hrana-macke.png?v=1754066639&width=600", text: "Раціони за віком і потребами" },
    { name: "Purina Pro Plan", image: "https://www.purina.com/sites/default/files/products/2023-05/Pro-Plan-Adult-Salmon-Rice-Formula-Dry-Cat-Food.png", text: "Корми для котів і собак" },
    { name: "Hill’s Science Diet", image: "https://cdn11.bigcommerce.com/s-xfu1s3ki5p/images/stencil/700w/attribute_rule_images/4459_source_1772138542.png", text: "Лінійки повсякденного харчування" },
    { name: "Brit Care", image: "https://brit-petfood.com/assets/images/products/cat/sterilized.jpg", text: "Корм і ласощі для улюбленців" },
  ];
  return <main className="container section-space"><span className="section-overline">Виробники</span><h1>Бренди зоотоварів</h1><p className="mb-7">Перейдіть до товарів бренду у каталозі. Наявність залежить від локального каталогу.</p><div className="products-grid">{brands.map((brand) => <Link className="product-card p-5" key={brand.name} to={`/catalog?brand=${encodeURIComponent(brand.name)}`}><div className="h-44 flex items-center justify-center"><img className="max-h-full max-w-full object-contain" src={brand.image} alt={brand.name} loading="lazy" /></div><h2 className="mt-4 text-lg font-bold">{brand.name}</h2><p>{brand.text}</p><span className="text-link">Дивитися товари <Icon name="arrow" size={15} /></span></Link>)}</div></main>;
}

export default App;
