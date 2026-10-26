// SPDX-License-Identifier: GPL-3.0-or-later AND BSD-3-Clause AND PSF-2.0
// Upstream MT19937/CPython notices: ../../../third_party/ and ../../../NOTICE.
//! CPU Pac-Man simulator with CPython-compatible integer-seeded MT19937.
//! ABI is C-compatible; no Python/Rust packages required at Rust build time.
//! Entry points require valid buffers of the documented lengths.
use std::ffi::{c_char, CStr};
use std::ptr;

const N: usize = 624;
const UNREACHABLE: usize = usize::MAX;

struct PythonMt {
    state: [u32; N],
    index: usize,
}
impl PythonMt {
    fn new(seed: u64) -> Self {
        let mut mt = Self {state: [0; N], index: N};
        mt.seed(seed);
        mt
    }
    fn seed(&mut self, seed: u64) {
        self.state[0] = 19650218;
        for i in 1..N {
            let p = self.state[i - 1];
            self.state[i] = 1812433253u32.wrapping_mul(p ^ (p >> 30)).wrapping_add(i as u32);
        }
        // CPython Random.seed(int) uses the little-endian 32-bit limbs of |seed|.
        let key: Vec<u32> = if seed > u32::MAX as u64 {
            vec![seed as u32, (seed >> 32) as u32]
        } else {vec![seed as u32]};
        let mut i = 1usize;
        let mut j = 0usize;
        for _ in 0..N.max(key.len()) {
            let p = self.state[i - 1];
            self.state[i] = (self.state[i] ^ (p ^ (p >> 30)).wrapping_mul(1664525))
                .wrapping_add(key[j]).wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {self.state[0] = self.state[N - 1]; i = 1;}
            if j >= key.len() {j = 0;}
        }
        for _ in 0..(N - 1) {
            let p = self.state[i - 1];
            self.state[i] = (self.state[i] ^ (p ^ (p >> 30)).wrapping_mul(1566083941))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= N {self.state[0] = self.state[N - 1]; i = 1;}
        }
        self.state[0] = 0x80000000;
        self.index = N;
    }
    fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            for k in 0..N {
                let y = (self.state[k] & 0x80000000) | (self.state[(k + 1) % N] & 0x7fffffff);
                self.state[k] = self.state[(k + 397) % N] ^ (y >> 1)
                    ^ if y & 1 != 0 {0x9908b0df} else {0};
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }
    fn random(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as u64;
        let b = (self.next_u32() >> 6) as u64;
        ((a * 67108864 + b) as f64) / 9007199254740992.0
    }
    fn randbelow(&mut self, n: usize) -> usize {
        assert!(n > 0 && n <= u32::MAX as usize);
        let bits = (usize::BITS - n.leading_zeros()) as u32;
        loop {
            let value = (self.next_u32() >> (32 - bits)) as usize;
            if value < n {return value;}
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct NativeStats {
    pub steps: u32,
    pub collected: u32,
    pub pellets_left: u32,
    pub initial_pellets: u32,
    pub wall_bumps: u32,
    pub stationary_steps: u32,
    pub stall_streak: u32,
    pub max_stall_streak: u32,
    pub reversals: u32,
    pub ghost_count: u32,
    pub won: u8,
    pub done: u8,
    pub _padding: [u8; 2],
}

struct Game {
    h: usize,
    w: usize,
    walls: Vec<bool>,
    initial_pellets: Vec<bool>,
    pellets: Vec<bool>,
    neighbors: Vec<[usize; 4]>,
    player_initial: usize,
    player: usize,
    ghost_initial: Vec<usize>,
    ghosts: Vec<usize>,
    ghost_dirs: Vec<i8>,
    ghost_count: usize,
    ghost_every: u32,
    max_steps: u32,
    stats: NativeStats,
    previous_player: Option<usize>,
    rng: PythonMt,
}
impl Game {
    fn from_layout(layout: &str, ghosts: usize, ghost_every: u32, max_steps: u32, seed: u64) -> Option<Self> {
        let lines: Vec<&str> = layout.lines().collect();
        let h = lines.len();
        let w = lines.first()?.len();
        if h == 0 || w == 0 || h*w > 16384 || lines.iter().any(|line| line.len() != w) || ghosts > 2 || ghost_every == 0 {return None;}
        let mut walls = vec![false; h*w];
        let mut initial_pellets = vec![false; h*w];
        let mut player_initial = None;
        let mut ghost_initial = Vec::new();
        for (r, line) in lines.iter().enumerate() {
            for (c, tile) in line.bytes().enumerate() {
                let pos = r*w + c;
                match tile {
                    b'#' => walls[pos] = true,
                    b'.' => initial_pellets[pos] = true,
                    b'P' => player_initial = Some(pos),
                    b'G' => ghost_initial.push(pos),
                    _ => return None,
                }
            }
        }
        let player_initial = player_initial?;
        if ghosts > ghost_initial.len() {
            if ghost_initial.is_empty() {return None;}
            let mut furthest: Option<(usize, usize)> = None;
            for pos in 0..h*w {
                if walls[pos] || pos == player_initial {continue;}
                let distance = ghost_initial.iter().map(|old| (pos/w).abs_diff(old/w) + (pos%w).abs_diff(old%w)).min().unwrap();
                if furthest.map_or(true, |(_, prev)| distance > prev) {
                    furthest = Some((pos, distance));
                }
            }
            ghost_initial.push(furthest?.0);
        }
        let mut neighbors = vec![[UNREACHABLE; 4]; h*w];
        for pos in 0..h*w {
            if walls[pos] {continue;}
            let r = pos/w;
            let c = pos%w;
            if r>0 && !walls[pos-w] {neighbors[pos][0] = pos-w;}
            if r+1<h && !walls[pos+w] {neighbors[pos][1] = pos+w;}
            if c>0 && !walls[pos-1] {neighbors[pos][2] = pos-1;}
            if c+1<w && !walls[pos+1] {neighbors[pos][3] = pos+1;}
        }
        let mut game = Self { h,w,walls,initial_pellets:initial_pellets.clone(),pellets:initial_pellets,
            neighbors,player_initial,player:player_initial,ghost_initial,ghosts:Vec::new(),
            ghost_dirs:Vec::new(),ghost_count:ghosts, ghost_every,
            max_steps:if max_steps == 0 {(h*w*8) as u32} else {max_steps},
            stats:NativeStats::default(),previous_player:None,rng:PythonMt::new(seed) };
        game.reset(Some(seed), ghosts as u32, ghost_every);
        Some(game)
    }
    fn reset(&mut self, seed: Option<u64>, ghosts: u32, ghost_every: u32) {
        if let Some(seed) = seed {self.rng.seed(seed);}
        self.player = self.player_initial;
        self.ghost_count = (ghosts as usize).min(self.ghost_initial.len());
        self.ghosts = self.ghost_initial.iter().take(self.ghost_count).copied().collect();
        self.ghost_dirs = vec![-1; self.ghost_count];
        self.ghost_every = ghost_every;
        self.pellets.clone_from(&self.initial_pellets);
        self.previous_player = None;
        self.stats = NativeStats {initial_pellets:self.initial_pellets.iter().filter(|x|**x).count() as u32,
            pellets_left:self.initial_pellets.iter().filter(|x|**x).count() as u32,
            ghost_count:self.ghost_count as u32,..NativeStats::default()};
    }
    fn frame_len(&self) -> usize {4*self.h*self.w}
    fn observe(&self, frame: &mut [u8]) {
        let sz = self.h*self.w;
        frame.fill(0);
        for i in 0..sz {
            if self.walls[i] {frame[i] = 1;}
            if self.pellets[i] {frame[sz+i] = 1;}
        }
        frame[2*sz + self.player] = 1;
        for &g in &self.ghosts {frame[3*sz+g] = 1;}
    }
    fn step(&mut self, action: u8) -> Result<(f64, u8, u8), ()> {
        if self.stats.done != 0 || action > 3 {return Err(());}
        self.stats.steps += 1;
        let mut reward = -0.01;
        let old_player = self.player;
        let dest = self.neighbors[self.player][action as usize];
        if dest == UNREACHABLE {reward -= 0.1; self.stats.wall_bumps += 1;}
        else {self.player = dest;}
        if self.player == old_player {
            self.stats.stationary_steps += 1;
            self.stats.stall_streak += 1;
            self.stats.max_stall_streak = self.stats.max_stall_streak.max(self.stats.stall_streak);
        } else {
            self.stats.stall_streak = 0;
            if self.previous_player == Some(self.player) {self.stats.reversals += 1;}
        }
        self.previous_player = Some(old_player);
        if self.ghosts.contains(&self.player) {self.stats.done = 1; reward -= 5.0;}
        if self.stats.done == 0 && self.pellets[self.player] {
            self.pellets[self.player] = false;
            self.stats.collected += 1;
            self.stats.pellets_left -= 1;
            reward += 1.0;
        }
        if self.stats.done == 0 && self.stats.pellets_left == 0 {
            self.stats.done = 1;
            self.stats.won = 1;
            reward += 10.0;
        }
        if self.stats.done == 0 && !self.ghosts.is_empty() && self.stats.steps % self.ghost_every == 0 {
            let prior = self.ghosts.clone();
            for i in 0..self.ghosts.len() {
                let neighbors = self.neighbors[self.ghosts[i]];
                let mut available = [(0usize, 0usize);4];
                let mut count = 0;
                let mut forward: Option<(usize, usize)> = None;
                for (dir, pos) in neighbors.iter().enumerate() {
                    if *pos != UNREACHABLE {
                        available[count] = (dir,*pos);
                        count += 1;
                        if self.ghost_dirs[i] == dir as i8 {forward = Some((dir,*pos));}
                    }
                }
                if count == 0 {continue;}
                let (dir,pos) = if let Some(m) = forward {
                    if self.rng.random() < 0.7 {m}
                    else {available[self.rng.randbelow(count)]}
                } else {available[self.rng.randbelow(count)]};
                self.ghost_dirs[i] = dir as i8;
                self.ghosts[i] = pos;
            }
            if self.ghosts.contains(&self.player) || self.ghosts.iter().zip(prior.iter()).any(|(after, before)| *before == self.player && *after == old_player) {
                self.stats.done = 1;
                reward -= 5.0;
            }
        }
        let truncated = self.stats.steps >= self.max_steps && self.stats.done == 0;
        if truncated {self.stats.done = 1;}
        Ok((reward, (self.stats.done != 0 && !truncated) as u8, truncated as u8))
    }
}

pub struct Batch { games: Vec<Game>, frame_len: usize }

#[no_mangle]
pub unsafe extern "C" fn pacman_create(layout: *const c_char, count: u32, ghosts: u32,
                                          ghost_every: u32, max_steps: u32, base_seed: u64) -> *mut Batch {
    if layout.is_null() || count == 0 || count > 4096 {return ptr::null_mut();}
    let text = match CStr::from_ptr(layout).to_str() {Ok(x)=>x,Err(_)=>return ptr::null_mut()};
    let mut games = Vec::with_capacity(count as usize);
    for i in 0..count {
        match Game::from_layout(text, ghosts as usize, ghost_every, max_steps, base_seed.wrapping_add(i as u64)) {
            Some(g)=> games.push(g),None=>return ptr::null_mut(),
        }
    }
    let frame_len = games[0].frame_len();
    Box::into_raw(Box::new(Batch {games,frame_len}))
}

#[no_mangle]
pub unsafe extern "C" fn pacman_destroy(ptr: *mut Batch) {
    if !ptr.is_null() {drop(Box::from_raw(ptr));}
}

#[no_mangle]
pub unsafe extern "C" fn pacman_reset_one(ptr: *mut Batch, index: u32, seed: u64, ghosts: u32,
                                            ghost_every: u32, frame: *mut u8, stats: *mut NativeStats) -> i32 {
    if ptr.is_null() || frame.is_null() || stats.is_null() || ghosts > 2 || ghost_every == 0 {return -1;}
    let batch = &mut *ptr;
    let game = match batch.games.get_mut(index as usize) {Some(g)=>g,None=>return -2};
    if ghosts as usize > game.ghost_initial.len() {return -3;}
    game.reset(if seed == u64::MAX {None} else {Some(seed)}, ghosts, ghost_every);
    game.observe(std::slice::from_raw_parts_mut(frame, batch.frame_len));
    *stats = game.stats;
    0
}

#[no_mangle]
pub unsafe extern "C" fn pacman_observe(ptr: *const Batch, frames: *mut u8, stats: *mut NativeStats) -> i32 {
    if ptr.is_null() || frames.is_null() || stats.is_null() {return -1;}
    let batch = &*ptr;
    for (i,game) in batch.games.iter().enumerate() {
        let offset = i*batch.frame_len;
        game.observe(std::slice::from_raw_parts_mut(frames.add(offset),batch.frame_len));
        *stats.add(i) = game.stats;
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn pacman_step(ptr: *mut Batch, actions: *const u8, frames: *mut u8,
                                        rewards: *mut f64, terminated: *mut u8, truncated: *mut u8,
                                        stats: *mut NativeStats) -> i32 {
    if ptr.is_null() || actions.is_null() || frames.is_null() || rewards.is_null() || terminated.is_null() || truncated.is_null() || stats.is_null() {return -1;}
    let batch = &mut *ptr;
    // Avoid partially updated batch on an invalid action or a finished episode.
    for (i,game) in batch.games.iter().enumerate() {
        if *actions.add(i)>3 || game.stats.done!=0 {return -2;}
    }
    for (i,game) in batch.games.iter_mut().enumerate() {
        let (reward, term, trunc) = match game.step(*actions.add(i)) {Ok(x)=>x,Err(_)=>return -3};
        game.observe(std::slice::from_raw_parts_mut(frames.add(i*batch.frame_len),batch.frame_len));
        *rewards.add(i) = reward;
        *terminated.add(i) = term;
        *truncated.add(i) = trunc;
        *stats.add(i) = game.stats;
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mt_python_seed() {
        let mut mt=PythonMt::new(42);
        assert_eq!(mt.random(),0.6394267984578837);
        assert_eq!(mt.random(),0.025010755222666936);
    }
    #[test]
    fn simple_game() {
        let mut g=Game::from_layout("#####\n#P.G#\n#####",1,1,0,42).unwrap();
        assert_eq!(g.stats.pellets_left,1);
        assert_eq!(g.step(3).unwrap().0,10.99);
    }
}
