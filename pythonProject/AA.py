#NLGN WT vs HEM
filepath = r'/Volumes/Wilbrecht_file_server/albert ader/rotarodweight.csv'
df = pd.read_csv(filepath)

# Function to plot WT and HEM performance with error bars and mean
def plot_wt_vs_hem_performance(df, wt_ids, hem_ids):
    plt.figure(figsize=(10, 6))

    def plot_group(group_ids, color_individual, color_mean, label_prefix):
        group_data = []

        for mouse_id in group_ids:
            df_mouse = df[df['asd_id'] == mouse_id].copy()

            df_mouse['trial'] = pd.to_numeric(df_mouse['trial'], errors='coerce')
            df_mouse['performance'] = pd.to_numeric(df_mouse['performance'], errors='coerce')
            df_mouse = df_mouse.sort_values('trial')

            if len(df_mouse) == 0:
                continue

            plt.plot(df_mouse['trial'], df_mouse['performance'],
                     color=color_individual, alpha=0.5)  # No label here
            group_data.append(df_mouse.set_index('trial')['performance'])

        if group_data:
            df_group = pd.concat(group_data, axis=1)
            group_mean = df_group.mean(axis=1)
            group_sem = df_group.sem(axis=1)

            plt.errorbar(group_mean.index, group_mean.values, yerr=group_sem.values,
                         color=color_mean, label=f'{label_prefix} Mean ± SEM',
                         linewidth=2, capsize=3)

    # Plot WT
    plot_group(wt_ids, color_individual='gray', color_mean='black', label_prefix='WT')

    # Plot HEM
    plot_group(hem_ids, color_individual='lightgreen', color_mean='darkgreen', label_prefix='HEM')

    # p value
    from scipy.stats import ttest_ind
    # Gather mean performances across trials per animal
    wt_means = []
    for mouse_id in wt_ids:
        df_mouse = df[df['asd_id'] == mouse_id]
        perf = pd.to_numeric(df_mouse['performance'], errors='coerce')
        if len(perf.dropna()) > 0:
            wt_means.append(perf.mean())
    hem_means = []
    for mouse_id in hem_ids:
        df_mouse = df[df['asd_id'] == mouse_id]
        perf = pd.to_numeric(df_mouse['performance'], errors='coerce')
        if len(perf.dropna()) > 0:
            hem_means.append(perf.mean())
    # Perform unpaired t-test
    t_stat, p_val = ttest_ind(wt_means, hem_means, equal_var=False)
    # Annotate the plot with the p-value
    plt.text(2, 275, f'p = {p_val:.3f}', ha='center', fontsize=12)
    # Formatting
    plt.title('Rotarod Performance: WT vs HEM')
    plt.xlabel('Trial Number')
    plt.ylabel('Latency to Fall (seconds)')
    plt.xticks(range(1, 13))
    plt.ylim(0, 300)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

wt_ids = ['ASD427', 'ASD428', 'ASD429', 'ASD430', 'ASD434', 'ASD457', 'ASD458', 'ASD460', 'ASD474', 'ASD477']
hem_ids = ['ASD436', 'ASD459', 'ASD462', 'ASD463', 'ASD464', 'ASD465', 'ASD470', 'ASD471', 'ASD472', 'ASD473', 'ASD475', 'ASD476']

plot_wt_vs_hem_performance(df, wt_ids, hem_ids)

#ANOVA on Nlgn genotype and weight

filepath = r'/Volumes/Wilbrecht_file_server/albert ader/rotarodweight.csv'
df = pd.read_csv(filepath)

# Define your animal groups
wt_ids = ['ASD427', 'ASD428', 'ASD429', 'ASD430', 'ASD434', 'ASD457', 'ASD458', 'ASD460', 'ASD474', 'ASD477']
hem_ids = ['ASD436', 'ASD459', 'ASD462', 'ASD463', 'ASD464', 'ASD465', 'ASD470', 'ASD471', 'ASD472', 'ASD473', 'ASD475', 'ASD476']

# Add a 'genotype' column based on ASD ID
df['genotype'] = df['asd_id'].apply(lambda x: 'WT' if x in wt_ids else 'HEM' if x in hem_ids else None)

# Remove rows without valid genotype
df = df[df['genotype'].notna()]

# Convert necessary columns to numeric
df['performance'] = pd.to_numeric(df['performance'], errors='coerce')
df['weight'] = pd.to_numeric(df['weight'], errors='coerce')

# Drop any rows with missing data in key columns
df = df.dropna(subset=['performance', 'weight'])

# Ensure 'genotype' is treated as a categorical variable
df['genotype'] = df['genotype'].astype('category')

# Run two-way ANOVA with interaction between genotype and weight
model = smf.ols('performance ~ genotype * weight', data=df).fit()

# Print the full summary of the regression
print(model.summary())