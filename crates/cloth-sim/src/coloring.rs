use std::collections::HashMap;
use std::ops::Range;

/// Coloração gulosa: itens com a mesma cor não compartilham nenhuma partícula,
/// logo podem ser resolvidos em paralelo sem condição de corrida.
///
/// Usa uma máscara `u128` por partícula (até 128 cores; malhas triangulares precisam de ~10).
pub fn greedy_coloring<const N: usize>(items: &[[u32; N]], particle_count: usize) -> Vec<u32> {
    // Mantém um bitset contíguo e rápido para as primeiras 128 cores. Só vértices de
    // grafos degenerados que excedem esse limite recebem palavras adicionais no mapa.
    let mut used = vec![0u128; particle_count];
    let mut overflow = HashMap::<usize, Vec<u64>>::new();
    let mut colors = Vec::with_capacity(items.len());
    for item in items {
        let mut color = 0usize;
        loop {
            let available = if color < 128 {
                let bit = 1u128 << color;
                item.iter().all(|&particle| used[particle as usize] & bit == 0)
            } else {
                let extra_color = color - 128;
                let word = extra_color / 64;
                let bit = 1u64 << (extra_color % 64);
                item.iter().all(|&particle| {
                    overflow.get(&(particle as usize))
                        .and_then(|words| words.get(word))
                        .is_none_or(|mask| mask & bit == 0)
                })
            };
            if available { break; }
            color += 1;
        }

        if color < 128 {
            let bit = 1u128 << color;
            for &particle in item {
                used[particle as usize] |= bit;
            }
        } else {
            let extra_color = color - 128;
            let word = extra_color / 64;
            let bit = 1u64 << (extra_color % 64);
            for &particle in item {
                let words = overflow.entry(particle as usize).or_default();
                if words.len() <= word {
                    words.resize(word + 1, 0);
                }
                words[word] |= bit;
            }
        }
        colors.push(color as u32);
    }
    colors
}

/// Reordena `items` agrupando por cor (estável) e devolve as faixas contíguas de cada cor.
pub fn group_by_color<T: Copy>(items: &[T], colors: &[u32]) -> (Vec<T>, Vec<Range<usize>>) {
    debug_assert_eq!(items.len(), colors.len());
    let color_count = colors.iter().copied().max().map_or(0, |c| c as usize + 1);
    let mut counts = vec![0usize; color_count];
    for &c in colors {
        counts[c as usize] += 1;
    }

    let mut ranges = Vec::with_capacity(color_count);
    let mut offsets = Vec::with_capacity(color_count);
    let mut start = 0;
    for &n in &counts {
        ranges.push(start..start + n);
        offsets.push(start);
        start += n;
    }

    let mut order = vec![0usize; items.len()];
    for (i, &c) in colors.iter().enumerate() {
        order[offsets[c as usize]] = i;
        offsets[c as usize] += 1;
    }
    let sorted = order.into_iter().map(|i| items[i]).collect();
    (sorted, ranges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coloring_separates_shared_particles() {
        // cadeia 0-1, 1-2, 2-3: arestas vizinhas precisam de cores diferentes
        let items = [[0u32, 1], [1, 2], [2, 3]];
        let colors = greedy_coloring(&items, 4);
        assert_ne!(colors[0], colors[1]);
        assert_ne!(colors[1], colors[2]);
    }

    #[test]
    fn coloring_supports_more_than_128_conflicting_items_without_panicking() {
        let items = vec![[0u32]; 130];
        let colors = greedy_coloring(&items, 1);
        assert_eq!(colors.len(), 130);
        assert_eq!(colors.iter().copied().collect::<std::collections::HashSet<_>>().len(), 130);
        assert_eq!(colors.iter().copied().max(), Some(129));
    }

    #[test]
    fn grouping_keeps_all_items() {
        let items = [10, 20, 30, 40];
        let colors = [1u32, 0, 1, 0];
        let (sorted, ranges) = group_by_color(&items, &colors);
        assert_eq!(sorted, vec![20, 40, 10, 30]);
        assert_eq!(ranges, vec![0..2, 2..4]);
    }
}
