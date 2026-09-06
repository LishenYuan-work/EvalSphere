"use client";
import { createContext, useContext, useEffect, useState } from "react";
import { api, type UserProfile } from "./api";
import { supabase } from "./supabase";

type AuthState = { user: UserProfile | null; loading: boolean; login: (email: string, password: string) => Promise<void>; register: (data: Parameters<typeof api.auth.register>[0]) => Promise<void>; guestLogin: () => Promise<void>; logout: () => void; };
const Context = createContext<AuthState>({ user: null, loading: true, login: async () => {}, register: async () => {}, guestLogin: async () => {}, logout: () => {} });
export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<UserProfile | null>(null); const [loading, setLoading] = useState(true);
  useEffect(() => {
    let mounted = true;
    const hydrate = async () => {
      try {
        if (supabase) {
          const { data } = await supabase.auth.getSession();
          if (data.session) {
            const result = await api.auth.supabaseExchange({ access_token: data.session.access_token });
            if (mounted) setUser(result.user);
            return;
          }
        }
        const result = await api.auth.me().catch(async () => {
          await api.auth.refresh();
          return api.auth.me();
        });
        if (mounted) setUser(result);
      } catch {
        if (mounted) setUser(null);
      } finally {
        if (mounted) setLoading(false);
      }
    };
    void hydrate();
    if (!supabase) return () => { mounted = false; };
    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      if (!session) {
        if (mounted) setUser(null);
        return;
      }
      // Defer the exchange so Supabase can finish its internal auth transaction.
      setTimeout(() => {
        void api.auth.supabaseExchange({ access_token: session.access_token })
          .then((result) => { if (mounted) setUser(result.user); })
          .catch(() => { if (mounted) setUser(null); })
          .finally(() => { if (mounted) setLoading(false); });
      }, 0);
    });
    return () => { mounted = false; listener.subscription.unsubscribe(); };
  }, []);
  async function login(email: string, password: string) {
    if (supabase) {
      const result = await api.auth.supabaseLogin({ email, password, remember_me: true });
      setUser(result.user);
      return;
    }
    const result = await api.auth.login({ email, password, remember_me: true }); setUser(result.user);
  }
  async function register(data: Parameters<typeof api.auth.register>[0]) {
    if (supabase) {
      const { data: result, error } = await supabase.auth.signUp({
        email: data.email,
        password: data.password,
        options: {
          emailRedirectTo: `${window.location.origin}/auth/callback?next=/`,
          data: { display_name: data.display_name, organization_name: data.organization_name, invite_token: data.invite_token },
        },
      });
      if (error) throw new Error(error.message);
      if (!result.session) throw new Error("注册成功，请查收 Supabase 验证邮件；验证后即可登录。");
      const exchanged = await api.auth.supabaseExchange({ access_token: result.session.access_token, ...data });
      setUser(exchanged.user);
      return;
    }
    const result = await api.auth.register(data); if (!result.user.email_verified) throw new Error("注册成功，请先验证邮箱后再登录。验证链接已发送到您的邮箱。"); setUser(result.user);
  }
  async function guestLogin() { const result = await api.auth.guest(); setUser(result.user); }
  function logout() { if (user?.is_guest) { api.auth.logout().catch(() => undefined); } else { void supabase?.auth.signOut(); api.auth.logout().catch(() => undefined); } setUser(null); }
  return <Context.Provider value={{ user, loading, login, register, guestLogin, logout }}>{children}</Context.Provider>;
}
export const useAuth = () => useContext(Context);
