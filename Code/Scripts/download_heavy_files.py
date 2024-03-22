import os
import requests
from tqdm import tqdm
import shutil

def download_file(url, save_path):
    """
    Downloads a file from the given URL and saves it to the specified path.

    Args:
    - url (str): The URL from which to download the file.
    - save_path (str): The path where the downloaded file will be saved.
    """

    print('\nStarting the download\n')
    
    # Request the file from the URL
    response = requests.get(url, stream=True)
    
    # Get the total size of the file
    total_size = int(response.headers.get('content-length', 0))
    
    # Set the block size for downloading
    block_size = 1024  # 1 Kibibyte

    # Initialize tqdm progress bar
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

    # Open the file and start downloading
    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    # Close the progress bar
    progress_bar.close()

    # Check if download was successful
    if total_size != 0 and progress_bar.n != total_size:
        print("Download failed. Please try again.")
    else:
        print(f"File downloaded successfully and saved at {save_path}")



if __name__ == "__main__":

    # Define the main URL from which files will be downloaded
    MAIN_URL = 'https://bertagnoli.ddns.net/static/PublicDrive'

    CURRENT_DIR_PATH = os.path.dirname(__file__)

    # Dictionary containing files to be downloaded along with their save paths
    files_to_download = {
        '1.blend': os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'BlenderScenes'),
        '2.blend': os.path.join(CURRENT_DIR_PATH, '..', '..', 'Data', 'BlenderScenes'),
        'Blender-2.79b-Linux.tar.xz': os.path.join(CURRENT_DIR_PATH, '..', '..'),
        'Blender-2.79b-Windows.zip': os.path.join(CURRENT_DIR_PATH, '..', '..'),
        'YCB': os.path.join(CURRENT_DIR_PATH, '..', '..', 'data', 'YCB')
    }

    # Loop through the files to download
    for file, save_path in files_to_download.items():
        while True:
            choice = input(f'\nDo you want to download {file}? [y]/n\n')
            if choice == 'n' or choice == 'y' or choice == '':
                break
            else:
                print('\nType y for yes, n for no!')
        
        choice = str(choice).lower()
        if choice == 'y' or choice == '':

            # Create directories if they don't exist
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            # Remove old file if exists
            if os.path.exists(os.path.join(save_path, file)):
                print(f'Removing old {os.path.join(save_path, file)}')
                os.remove(os.path.join(save_path, file))

            # Download the file
            download_file(MAIN_URL + f'/{file}', os.path.join(save_path, file))

            # For Blender files, remove old Blender installation and unpack the new one
            if file == 'Blender-2.79b-Linux.tar.xz' or file == 'Blender-2.79b-Windows.zip':
                old_blender_folder = os.path.join(save_path, file[:-7])
                if os.path.exists(old_blender_folder):
                    print('Removing old blender')
                    shutil.rmtree(old_blender_folder)

                print('Unpacking blender')

                # Unpack the Blender file and remove the downloaded archive
                command = f'tar -xf {os.path.join(save_path, file)} -C {save_path}'
                os.system(command)
                os.remove(os.path.join(save_path, file))

            
            if file == 'YCB':

                # Create directories if they don't exist
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                # Download the file
                download_file('https://public.boxcloud.com/d/1/b1!nCeNYuaB4BPc8brMIeVCigiPFHlLLI7T9uxgf9_fQA0_8eIkaX375jQgnT9ogFC92X_3YJP8C8ZvI0YLswY1A4bRon3YmLcdgzsdNVMhJVHrwxLqQOlKnWxDms8IpztVPSQ5i6F8cO3Qg7J6eow-nKw87Gl5mK7MvFBPs6jJIoOuqhRduD1mtFLyS1JquB7ypBuiaBLEGl8mI5YXjNpcIfVA1-6sA-RQZ4EBeZ3w-sFGcn6korBOVoMrVb6M_F9lQMLdnS2qqeif86QnRmB8INY_8_gaRcQZAdFDhKLCYHGkppoLEh60rAIIhx8kv-9Abv2-4Ud80OuTzlXtTlvxhCEpZCcgGjCEIJQwUWItRQeETZp6-L_yINA_CphdXU4mc4QSnfUgsF0yJKwODIARKwf5OUOcqNoyl4ErNIxBMMflFQtO8UGb2KET-sXFYHGNzF4wOr_RDMOR6eO9BWK8BRS69NE5JN3ZTS8mG_tAWe0m9iDjjqLtMW-Yz03GhqbQG0hZnF2e8RHV5iRLUyHIe9IRX4cfXBPdZiUDnVLEVqpOG_F_7BX3NOfAUXRTVuxM96650BQu5GkvdDcu0ij8SjH_UhpnNK_3tthpB_YXgZC-mZyPlWRLZaw3DadNkj4Hmq-VMMDBaYfJ3YcqVwahq4l4tM3zxyosxLOIkc_RykKcQCNvJ2qwTZlxlwQnOf33cEcK53ufjOizcBL41HdypK2FZapgsviLJufJJtr_hwVTeoe5P8ftHe-nG3eiWUsf38Gr_0d3u3kcK7UzvYN2grtISMv5YphoeUfHzi2G7-KwRkWKzZbUYlj9k-WDdX3P_Gz1-MPT0N86ZVBCy9BONRhTtizbYXPer8AwV6Iae3m-XiiquDv2dtEKSTWObnNV4oCN4Grg5HwTc84Xuj1XGprs4iwJhuR4ejkvoj5WgMGUgjbh3g1h-Rq7gie3Cztdq7nvW9kTpoqMtGJmdzxp6r4gLLWEcIqjMrAskm8oD35jhfktBHuxuOj0henzEx9tUWIFfwYn_C-bq9To5dZbZKMBhvJCPaQEbaU5x1PV77WKluWnhrJxX_JtA11z4bqyW97PzxN63NA3wdMySfASkMgvTrU8BPFGtloyCkLktzJI1PyfbPKL15T4daBTmFAcdqjH5_2X1a-zYafK3Tokk3J_YbTQAWlzpynqNCfAu5SZKYQh7G5WYAVeN1Wzg8UHF06KLP8bSjDtK1szX2IQ_sTFnSbJRkQ./download', save_path)
                download_file('https://public.boxcloud.com/d/1/b1!wzO3Qlb7EkkshRb0zIKRS1ION1HG0r2edJAop7S_wrwno7W-KU7vluT3Xa3XE6RwHuTNk5lyG1j9tjnFsBgbcV7YJ4iIFgWzUHu3wv4r_YJwjml9qnTHPKgeljk_FG6hfzQvNGFVKFu5gnJc7jaJ6yUDAPyV2u4l3fjYl-GNUg3vCT4Kq8l4btcdiKAlIQNcbzH8EMeKBogaMgTxbfWcAg9Kh_X1KZ8Od1e62LIXSen_CAKRRs0de6XsukOp9tY5Kygwcsz5C6t2wBcxpBsU-KQ5foB3l8eihVK5I_FzpnstMoZcwXMCTP-1gEc6fnRtAhWEAg6GmWrHdfl65-miu_xDC5l2YrFoxhHoWpwXwTl6BKN0oABpyVlY2qYZQz5LCU0ETXXiFx9xsXCf_YFEdtaxjZaZ-QLuINknumCCasE1oFhZ5B7bvT6JcfAJ614nlA-oGfrQAMT3wwaFHnyi1SKt2zYXLvoPRFXlC46VjmwAfbMK-dmrXE5ygQQVsRgWduASTvAmDy3A_BgtyoVon0eeSFzQ5Bf8eWoPY53fI3jmX5oOO6DRjvlkvxAgTnP02hKWWOlGf7lZ77VbM4yuzmZi_BvDn-Mn5uQZGYgHrGsnKEzp02cBq5c_JRCDL_ckn84tPW0hWbUsNYH8LDSJNVXwea1X3GLV5e-dEVC1uLJFbc2eYmTRf__7jt26KnTyRKhNIN5GcambjNBEPLLdzI9o_U19nCats1leHROllQRG22VFvRpg9VXV_bO8kMvH7I66oLU0ohkLosHQBCnaGXzcIaoqjnoTrLIJrhRA93RpAe6e_vju-Z6XOY6cPTBxUbb_Yn_77QXgLbcdtLNSiaxqvTlOTHIAQxASehIoasTkaXlfA6zj3abtKHyafgle17nZENxZZ4wf7UwBxNDyl8r0oueK9MUVgfxRjuNGpgW6JKuQxEYVHCjUdCLQ7j6JDQDh380UXOVwF0sMaO1EwX9b2kV9sPEL86RPvsCW3VkEMkzH_BWFwQ_4sQZWIfEdCVSXuwXE_2Z-2OuJYCHJxETJLke17yZoIeAuI01caK8YT-t2fvHgxc6a-FnXnY-ByOS0UQDcv7aMDKKCjFQjCaxmY9D6WpEocoOoZDUrRs66esRSJKJ4-mGlVHMbOKlQgeDHnWJhdfIJomcIhdV2PPnku5pikorC9zHypWm4_5EXZRfV8qn2d5fPNN5LeAOPbqOjp4taZP1hIKE0Ocjsb9PTeZVfBVIG/download', save_path)
                download_file('https://public.boxcloud.com/d/1/b1!CssBJirnjMHFtg3qKTMHDVjMCu8nkMsN9vDp-GVOqc-ouipHrfuXKrNHgF3lIzoYLH2udRsyTdFVjzNXo0ConcXrvBboOgRhvUHgG264XTrdF_I50Axe8SpkoE3oDvuNZfhHhIZsfnwQtOhfn4vqQKbwira1HN6QNTTROxNy70qaxaGXH3QjxCcfDbYS0Nwh6_TLoODMmSuii-BBKzV7IaF-iEQUJwzHE-CNAepAZZuqBJlP0eciZsiKo7rp495pZXmVV2EX0zp7JzHQkHZCtLlI8NEPensoLOm3ytSnodG8zVFJIV5bARPlcsjjuVk-sy2KY9Vp_L2tJiszw4Nck2rFKn3Gr7Lu2fOE-6_ZRkh-NIGpEq7BJH7CREmZWLn5OfcQ_Hj5w5NxxowFCMUffW3HHLebM-3q2RPFni3Op77Z4NoAJcndp8ov-ZulPiARu4FtMdqJPnNJ07LGAla_1UvuBo_QRe6WLYRcedDy4baAeaeIw6bCV8G0jugZQomUxWE_bqna2MJ31GV0oujQWMAdI-r61uMym8AU-X0nx2V8dpqCiXSXn6-hgAUlAQJVXK96IzLv2RL2So2hP0ECBjn9fczW8P7AQ2Pmm-IzGZX6qX8BucRhyNTuqHNUKZRuSYxERKvJNLASP4-uyPYXjGBRYh4x2DUEjUriet4_zNorJ_Yb_6K-490RG5IVxvApU1LadPHb4BV5LXw5ugPM9CfbOGb7t2bA9kFEksFomsINDoSxS96yU9x9FdnBvem05Ko8fnv75v0aGDdra5UMT5j84ZHE7skU5mJ_Ou1mZT36Ax6_O2_KGWOuCjLgMiA89fqkgY25zFoQ7vxocyTpJRSfCG-eRoCTUJHYpXD3KA358wYt3y0wuLjyMxIMv9F1bD2JgnRFkn98CeqDfW2JebB73xXEc0TmxbN37SKWbv8J8BJsS3davMffgVDzrSisiy7VFZrmnxO-wyTAUXl3gE4o65IeVTHCFl1WuQyO3D8_RAUz-KPh6gZzykPPOD-rdBdSiM8Gss0WiehGCsbBc_jPHXkkpPJaPmv9WPEuC5CcL3yjkeXfG-VJ52A5PjcCCuLfNQO3zkUnYhO6LJa8lGss0Fm0neBElpR1cQyTMLMzQxcNoupil9vA0V8Ad_4scwNVORou2VvH3r0llt0ohw7XWSaQjdDml1N79g4AARsRZc0zNXXh2S6Ak8IHbhyWgktq3QozyC0wZ1qi4KjLS6eJaawq1F3E/download', save_path)
                download_file('https://public.boxcloud.com/d/1/b1!-vjuq0_f3AsEBFyf9rO5OaDpO69MhdZSsmyobsGGy1gz6CehGX97RrKNgk_gjpbCMv-xkaZHcqfYXPGjAdfqfQlZZLZgPR94QbbB_G6GxBSb35b6WYVkaj9m5EkbYzzjNSBEcG5APokYpj9KWrQGU6J1vKpyvZlNAH1l5G4Lb2eHA9k4EWOygnEA8AsqS5qECXFpAdMNM6TNimd59cyIwhNcecaA6EJqMSAcWMAPrD1BZX3sdGsqmnlW5KrzTndCQeGFXbOQv7EEQxORXq2dqx-twxLkrMlMjKJrZhUyoHnfTU4sqtLVpElF2wbkJiz1VFklAhfg0-DAMfUs_Wsy4ORPjzzqV7hHGsB29sUD76NrOJbp_sK-OOEsXfgkgxjKuPlLcaq1x6gn8YS01AdjiKkYi0FgrfI6bB3DfSJC0e9bzVWefoIqb4UvRhbQ13GCZdiYAx_BQ95PPKSxFUc4LMkJHg0D4jNhp8-JWsjgcgZh2roqhMXS5KyuQcxvXx5HL3w59gP0nHj6FnmRgfp3HnjSC4pHl8zpCFqAOE8biDXdmZXpC0A51zc9UTcYQAdoJPWOT6Tgf5feOUOkwztFmkffktg2WlyBgnymnELWkZyUQ7bsZuJ1woDg7AlEHKrpZyg53z2_XRnZZ0oZrsEoJ3geDTVihJzoXgWCFEUAexAjLkiPdOfbTQW0PRQu7CJVq3QqyMRhGNvPMIMR1amdHUJ5MXhU4IF2KKgg8JqRt5U6rnA3Yeov8wPg1JO_JoFZEF1DQzl90xWq1trFbmupj4pd3QMx3bCnO_mrT7EQC4R_MtYEfmBuK7kzxgIKTsnuy-5hLUTT8cnlSlfghO9Vgh8t0vfRpZIphl2vKIa7pq8k_sf6TYabHWdwO7itrw2bQv8t2WzRDCm_pVsj0oXDYc7kXyCjPC9SEMnzadrg2wHZzynZkHW-6u_Zrn3uA_3GjJgu7IrNHuDVM_rqBDSELo2UCzvkS9VZyLQuvCyGQ_cWT1km2ujvBJQhhvoVXineRimTakic2QGPgRwpUIz8D3SELGHOM-Kkk4AieI64aVL_XbN300MNgtVX-YGkBy08ebkcmkJXRsSF3oS0pi0W17o1rhuOiyYt0KpHZ94DhRd4NN6ln8Ko8fvxrLj5NVFHA21z8N3UHgWJSOyX0uXeVB96YcQ_CLKIYymAcnAO4_Y1KIn0zeExk5Z8UaHSvVpgUYfkDCz8km7bEkRcMQqFklwC0gqTU7JVPZP98UHg/download', save_path)
                
                
                command = f'tar -xf {os.path.join(save_path, "data1.zip")} -C {save_path}'
                os.system(command)
                command = f'tar -xf {os.path.join(save_path, "data2.zip")} -C {save_path}'
                os.system(command)
                command = f'tar -xf {os.path.join(save_path, "data3.zip")} -C {save_path}'
                os.system(command)
                command = f'tar -xf {os.path.join(save_path, "YCB-Video-Base.zip")} -C {save_path}'
                os.system(command)