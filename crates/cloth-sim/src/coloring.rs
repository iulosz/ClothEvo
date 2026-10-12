use std::ops::Range;

/// Coloração gulosa: itens com a mesma cor não compartilham nenhuma partícula,
/// logo podem ser resolvidos em paralelo sem condição de corrida.
///
/// Usa uma máscara `u128` por partícula (até 128 cores; malhas triangulares precisam de ~10).
pub fn greedy_coloring<const N: usize>(items: &[[u32; N]], particle_count: usize) -> Vec<u16> {
    // Bitsets dinâmicos mantêm o caminho rápido em malhas normais e evitam panic
    // quando uma entrada degenerada exige mais de 128 cores.
    let mut used = vec![Vec::<u64>::new(); particle_count];
    let mut colors = Vec::with_capacity(items.len());
    for item in items {
        let mut color = 0usize;
        loop {
            let word = color / 64;
            let bit = 1u64 << (color % 64);
            let available = item.iter().all(|&particle| {
                used[particle as usize].get(word).is_none_or(|mask| mask & bit == 0)
            });
            if available { break; }
            color += 1;
            assert!(color <= u16::MAX as usize, "excedido o limite representável de cores de restrição");
        }
        let word = color / 64;
        let bit = 1u64 << (color % 64);
        for &particle in item {
            if used[particle as usize].len() <= word {
                used[particle as usize].resize(word + 1, 0);
            }
            used[particle as usize][word] |= bit;
        }
        colors.push(color as u16);
    }
    colors
}

/// Reordena `items` agrupando por cor (estável) e devolve as faixas contíguas de cada cor.
pub fn group_by_color<T: Copy>(items: &[T], colors: &[u16]) -> (Vec<T>, Vec<Range<usize>>) {
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
        let colors = [1u16, 0, 1, 0];
        let (sorted, ranges) = group_by_color(&items, &colors);
        assert_eq!(sorted, vec![20, 40, 10, 30]);
        assert_eq!(ranges, vec![0..2, 2..4]);
    }
}
