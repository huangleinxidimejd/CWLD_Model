from matplotlib import pyplot as plt


Deeplabv3_F1 = [0.5259592242007829, 0.539409083917442, 0.5719312974884322, 0.48821518715241463, 0.5490594134403216, 0.5493792516278145, 0.5642064363017092, 0.5883869097902085, 0.5309095833608067, 0.5851363827531318, 0.5355262178371184, 0.7104880142560711, 0.5734050029768871, 0.6193887729902267, 0.6720169048502719, 0.6470859721041721, 0.6603400336075677, 0.7485175990855201, 0.6809957005583508, 0.6916745837903596, 0.6754777728705329, 0.6856519248408173, 0.6902872099948594, 0.7694269418572324, 0.7442189688855423, 0.7660340327652018, 0.770588233836127, 0.7957625993486027, 0.7383368282524371, 0.7899815041678748, 0.7616990858946093, 0.7878276482628787, 0.7454915081989988, 0.8063764656072699, 0.8024428748690744, 0.7852088486047981, 0.6923848189611485, 0.8148659271510056, 0.7767961564337393, 0.8242382758587422, 0.8269756260072451, 0.8189056234600897, 0.8031410354312905, 0.8068518372763568, 0.811461467195701, 0.7957691411937762, 0.8060184887879901, 0.8281345156481729, 0.8023060260987214, 0.8285149361660293, 0.8517250546102273, 0.825692262316228, 0.8173123429986445, 0.8437555724733311, 0.733017473122222, 0.8136076175507903, 0.8396638504353839, 0.8240078439037762, 0.8182126805280662, 0.8342926409477242, 0.8404514555656573, 0.8470470599722831, 0.8373574115757431, 0.8485534140868036, 0.8497911905740192, 0.8366981985423142, 0.8422358811936386, 0.7979797396555767, 0.8386407126989143, 0.8196183083188842, 0.8542253846236296, 0.8607332020823915, 0.8458465464941304, 0.8357915144981602, 0.855487801507984, 0.8256500969016306, 0.8488802987569616, 0.848177611271715, 0.8418743713843497, 0.8605679220924293, 0.8617310895895485, 0.8434229864018159, 0.843691806465527, 0.8118548828329987, 0.8160734819099204, 0.8471619137586339, 0.8626594960695133, 0.8564163360394773, 0.864084878511828, 0.7849274523410071, 0.8513436754369754, 0.8689446935928465, 0.8748323888673286, 0.861621949982253, 0.8552218662429141, 0.8655679682432355, 0.8711347611729088, 0.834861594023335, 0.8760480138326655, 0.8766615380134687, 0.870249514288199, 0.8395445935077688, 0.8629838888806075, 0.8714166765682949, 0.8728412069202327, 0.839572210946402, 0.8656369217777381, 0.8674341389635359, 0.8578855138315319, 0.8685928200636891, 0.863040304322613, 0.856791981005158, 0.85132110608685, 0.8682607209740564, 0.8679778217321164, 0.8501204049709908, 0.8669700691870363, 0.8412289361015916, 0.8739822294155494, 0.8765962090500509, 0.8565627160018741, 0.8712298144822226, 0.8746838083424462, 0.8647997877850989, 0.8816860960529879, 0.8830648424821357, 0.867763033369295, 0.8619108294063785, 0.8702343652755509, 0.8627525722744986, 0.8631109474552559, 0.8753082433241464, 0.858250556408544, 0.8769299716156244, 0.8775381996316323, 0.8746794607617002, 0.8625353528289932, 0.8791888333858585, 0.864000079392013, 0.8754864898419126, 0.8713609908754241, 0.8662698975829135, 0.8701990606615936, 0.8753721181938598, 0.8787284295443746, 0.857888534304889, 0.8752722961962341, 0.879073504149812, 0.8768400475174521, 0.877926393556958, 0.8728158974883348, 0.861555414630305, 0.8733771050430683, 0.8682841746582939, 0.885704404867023, 0.8704978106509117, 0.8867163225517395, 0.8803657461323631, 0.8786922073795934, 0.8835683808268322, 0.8734181186209924, 0.8705219729484094, 0.8710622865864945, 0.8809610838565329, 0.8829436580179396, 0.8823508344929154, 0.8592589023951409, 0.8693185519548607, 0.8819613611400342, 0.8800332617981635, 0.8847833193582706, 0.8455707959024555, 0.8847812007566509, 0.8599026130221057, 0.8569645471018388, 0.8712200764562575, 0.8520993880640073, 0.8678465773016563, 0.8733930251988582, 0.8805779551379882, 0.8571071451327164, 0.8794763889411447, 0.8823369122154991, 0.8832846638414065, 0.8812875504699103, 0.8584451821381236, 0.8617468285074681, 0.8794190578076643, 0.8216402510353599, 0.8771120867945335, 0.884519132196491, 0.8702989433327424, 0.8888848818022056, 0.8822510154233797, 0.8841413224263857, 0.883923006336506, 0.8758070996272287, 0.8574697665967322, 0.8868887954410258, 0.8845895623732347]


pspnet_f1=[0.01754678026664529, 0.342407523442605, 0.08639909014864929, 0.27066667165814373, 0.24273543922455157, 0.27665680868749454, 0.38676787830093984, 0.23238418769131328, 0.37397213168220456, 0.2266396976151994, 0.326155276316599, 0.3203590902914239, 0.2640329370976696, 0.42196971026684554, 0.3257859941351101, 0.34667994918066747, 0.3131749204135595, 0.26429992035734906, 0.3631385549469456, 0.4147039669566476, 0.4584416251494075, 0.34427600002241066, 0.430831477741263, 0.3761741111889792, 0.43768809951891613, 0.4688369144351691, 0.37349510727547, 0.24634440023822327, 0.3302262991637401, 0.46914456770180835, 0.30155196878878626, 0.3234944041707935, 0.5219045695837673, 0.47053771073057615, 0.48370534724434916, 0.5016143129057379, 0.42472046632824095, 0.49367777541921937, 0.38471878613254457, 0.44095246365988694, 0.18885578413823872, 0.2998276373086511, 0.5336314916961445, 0.53507941756473, 0.5497392319732833, 0.5366862744605563, 0.4961795023796358, 0.2983993101168828, 0.49172743301619726, 0.5548567053002879, 0.568092205639394, 0.5862925268364034, 0.5644194275427806, 0.568870402841959, 0.5449140808526595, 0.5848785754737111, 0.45227163068747245, 0.5478701233902489, 0.6163580656557169, 0.6132609325183352, 0.4131584858442763, 0.5528379367449678, 0.5640779723413453, 0.46506057644925103, 0.5969779623783481, 0.5573319949255722, 0.5964041151633285, 0.6123969522057982, 0.6298510349086082, 0.6288453544049709, 0.6337120442728436, 0.5982411104382629, 0.5708496645562067, 0.6593021077050849, 0.644165043404744, 0.6436216348559707, 0.6606441509981735, 0.6445086074246761, 0.6777213031518935, 0.6755906115409769, 0.6801598518554318, 0.6767957952652435, 0.6797200844546205, 0.6763114408686678, 0.6194650735563584, 0.7055931307834414, 0.7075103007556891, 0.612921563608626, 0.650640386984448, 0.6104667466151098, 0.5724389794590258, 0.7109023767451026, 0.6427841610085252, 0.6770426545433976, 0.6861623236832693, 0.689597803321126, 0.6946932044465428, 0.6300223622008727, 0.6900004983154221, 0.7272847709162826, 0.6997895784026728, 0.7312229647003233, 0.7177031905371748, 0.7087406954123964, 0.733605643987391, 0.7177238836340083, 0.6856836115222256, 0.7358182276856114, 0.7481566471895886, 0.6941181631825584, 0.7020591052544481, 0.7485684037356668, 0.715234851814771, 0.7487083513208558, 0.7360255973980921, 0.735859181335406, 0.7590459024956123, 0.6844618689264222, 0.6594803146039175, 0.7263087444940964, 0.7098700500349489, 0.7546154092227254, 0.7624234358507345, 0.7659121435007952, 0.7439207907072567, 0.7622369047734008, 0.7322478042867514, 0.7345019657073477, 0.7584086366537782, 0.7291058926893498, 0.6979427993972569, 0.7693934754635925, 0.7582076373508332, 0.769879100131224, 0.7727549639840644, 0.7810873427187958, 0.7786102010763278, 0.7748475157694494, 0.7767868466523836, 0.7724785210534924, 0.764749911011092, 0.7760829228587661, 0.7822168466553716, 0.7737758868835872, 0.7882182911296485, 0.784996345918637, 0.7718643268719019, 0.757433359831241, 0.7615300301410753, 0.7798081510713197, 0.7839717615365341, 0.7869015484086324, 0.7919145709460349, 0.7652184700922848, 0.7833898019030126, 0.7809314519931791, 0.7836069181166183, 0.7771253639223387, 0.7699540412364972, 0.7847609510229228, 0.7749050030499466, 0.786625398560595, 0.7738460627539968, 0.7894357962462552, 0.7816400554085453, 0.7853385462133693, 0.791048184083935, 0.7742810546556763, 0.7745320872165364, 0.7751051338679662, 0.785462454349105, 0.7823812514467464, 0.7851669850670896, 0.7902508550633056, 0.7824590613724705, 0.7808356207038041, 0.7806406119064847, 0.7792279563818023, 0.7940553439288497, 0.7928672346803199, 0.7892899948450861, 0.7787101116804775, 0.7816190418383083, 0.7768505880213891, 0.7939684340037034, 0.7938354969295643, 0.7844976234976546, 0.7852235388359802, 0.7801408314254669, 0.7954933992141727, 0.7877322032645714, 0.7862929336596038, 0.7821432428047492, 0.7877586905674577, 0.7937480301050329, 0.7810349250240175, 0.7910714087023305, 0.791510897923721, 0.7820449663528288, 0.7878614896037716]



segnet_F1=[0.1822503254727505, 0.264857928195089, 0.36708092213744636, 0.35373626825417265, 0.30030055814305234, 0.39514996894336923, 0.3050181406746079, 0.47503342632142787, 0.423811734381549, 0.4310216677090406, 0.463514518368362, 0.5000844701299718, 0.4205720419027831, 0.5470097320680973, 0.505661656252999, 0.5541540933273775, 0.573642685327815, 0.5467611869477939, 0.5619979695469519, 0.5430998403142666, 0.6455435383718355, 0.5585275757928964, 0.4732916529891822, 0.6013996265453925, 0.619127227590351, 0.671588880822083, 0.6885490207922774, 0.721304602399121, 0.7139374070090984, 0.6283553106099645, 0.7146424553113813, 0.7532435087649203, 0.7498834625913607, 0.6958208833408934, 0.7619335225398726, 0.7582468868158304, 0.7717536305300479, 0.773033552500566, 0.7558394803101197, 0.7906926686501253, 0.761444582649536, 0.7996986807633605, 0.7962355484087319, 0.8021120766260547, 0.8132031967568069, 0.8087450482861661, 0.8115611125539973, 0.8200564199669564, 0.7863403454283959, 0.8136552739560561, 0.8115045159615883, 0.8276412950030501, 0.8278153513123562, 0.8363667991086292, 0.8342113951364454, 0.7973112544266473, 0.840127735370365, 0.8369107841212677, 0.8316162626540682, 0.8416542347754803, 0.8441497816134219, 0.8493951107974201, 0.8444821218915878, 0.8502793568055727, 0.84556366387275, 0.8508031087495442, 0.845013404957568, 0.8520386085739041, 0.8548894009116692, 0.8593191792864452, 0.8577711798707885, 0.8603647009835602, 0.854315231643587, 0.8588292956023255, 0.8617442468013271, 0.8606857618565148, 0.8612859734508801, 0.8559914067062112, 0.8614062670673726, 0.8609187973621023, 0.8604625655282699, 0.8647587623152888, 0.8640383683694147, 0.8626894347145262, 0.8640900008807884, 0.8570399432504422, 0.86845206601049, 0.8689006630311642, 0.8674556685636994, 0.8668247134941435, 0.8722454504768711, 0.872618645848489, 0.8708994459363877, 0.8717856062005699, 0.872326779822512, 0.8671218056139052, 0.8730644585793669, 0.8731407255973271, 0.8730209638059518, 0.8752467485376597, 0.8720530705209035, 0.8754828841967182, 0.8765185800937373, 0.8750693793220434, 0.8755082654533575, 0.8768172788872959, 0.8769640042557657, 0.8776664826616405, 0.8788398749291949, 0.8749665224066329, 0.8802977862796691, 0.8789580576226744, 0.8794368884923597, 0.8804785609769885, 0.8795821696202099, 0.8805654160674914, 0.8783222033458408, 0.8816447915922292, 0.8809535913184151, 0.8808640577616148, 0.8815380732476222, 0.8818016523075107, 0.8808483303901561, 0.8826175121770888, 0.8827167738078078, 0.8835913425393017, 0.8832233359151558, 0.8837031233399492, 0.8832792093362921, 0.8832772638390347, 0.8836659264324653, 0.8841893654915367, 0.8847449260208193, 0.8840737744438121, 0.8842048631502275, 0.8839847303208292, 0.8810095375523581, 0.8854678247227181, 0.885666797440639, 0.8857037178492179, 0.8857699020504364, 0.8861654619416163, 0.886027863757279, 0.8858786552359671, 0.8851445173250692, 0.8864279710419128, 0.8861949360473077, 0.8861877345452762, 0.886722811689552, 0.887184963230511, 0.8860647196996926, 0.8869280357173323, 0.887421339348817, 0.8864269207340753, 0.8874630801361533, 0.8873275542785434, 0.887428554651668, 0.8878717286217886, 0.8870960904443934, 0.8879480385620663, 0.8868650632582551, 0.8883328617726516, 0.8877710011913355, 0.8881411521171194, 0.8878550748406175, 0.8878485749209188, 0.8877247715273134, 0.888619412159418, 0.888883603759338, 0.8884647478276547, 0.8889419196511982, 0.8886265841939097, 0.8894018995993934, 0.8887532401096235, 0.8892285042266243, 0.8893199321284898, 0.889680202345436, 0.8891241667567328, 0.8896558444532936, 0.8891584980954172, 0.8887740519886826, 0.8892809531149576, 0.8886830343734902, 0.8888298009477265, 0.8896687815987695, 0.8893726874530947, 0.8891978274869562, 0.8889275165369599, 0.8890234565663406, 0.8895373545510845, 0.8892286500540149, 0.8895617879355574, 0.8889154053434469, 0.8900677229128545, 0.889407694294368, 0.889851036474073, 0.8894040768557755, 0.8892346213336703, 0.8893925648629862, 0.8891631314512931]


uner_F1= [0.3279239466371352, 0.32126582121939096, 0.3691238909195407, 0.4283249730705761, 0.3937581802618164, 0.3532899655948533, 0.36616537811146194, 0.4801265092011997, 0.484305315104757, 0.5163466694773947, 0.4327943742532806, 0.5160468777969307, 0.4657397525221509, 0.4931286822395855, 0.46568323939321243, 0.4539932860979053, 0.5084554618045863, 0.5139347599376856, 0.5656880512620727, 0.5843409367138547, 0.5880423169343446, 0.6162785378419189, 0.6292662638391782, 0.5265549754013331, 0.6271931085201474, 0.6240934440470663, 0.656623364077498, 0.4629159273066076, 0.6218853201267883, 0.6778056409013995, 0.6265826718140569, 0.5615137426631683, 0.5253887489126344, 0.7208768267148399, 0.6871935667319687, 0.7316027806123916, 0.7355773738900151, 0.6588458295059373, 0.7341313018807493, 0.739539137161244, 0.7527074926530293, 0.7505868131205501, 0.7057725162408839, 0.780974459428785, 0.7870973735993629, 0.7853381827110452, 0.7953502936202018, 0.7219259827327612, 0.8129066409605492, 0.6197667358741353, 0.7918632375786758, 0.8203523352080747, 0.7175023705415169, 0.8217660070970034, 0.8210913953022774, 0.7941840831256701, 0.831751651462859, 0.7664610673539162, 0.8010162221018705, 0.8026834895459213, 0.8355882197947787, 0.8400454242795463, 0.8131873375083185, 0.8465445968907159, 0.8514218412127524, 0.8419576268650688, 0.8468131700357848, 0.8536738298322203, 0.8574688769310173, 0.8547633583790981, 0.7772121401379407, 0.8536963946512776, 0.8558022353018312, 0.8585009757328248, 0.8575913873052516, 0.8586598730064002, 0.8650145568060088, 0.8518620089661719, 0.8624052586660648, 0.8549657873788145, 0.8593148817236214, 0.8660588429685129, 0.8599112442497052, 0.8581643988407146, 0.8672429382565753, 0.8671864379015211, 0.8591430358957517, 0.8664935661146255, 0.8693097821572829, 0.8723728323273314, 0.8695253647157509, 0.8701094620938481, 0.8718968385439835, 0.8542547188038466, 0.8614325499462078, 0.8711048916876013, 0.8666403415504197, 0.8727694846363958, 0.8753756842206021, 0.8108115303187537, 0.8756977465492426, 0.8723738490891553, 0.8733234845253534, 0.8754599277830973, 0.8761649119678567, 0.8777963968264304, 0.8762804235323275, 0.8746681166942514, 0.8772932675925172, 0.8767696091877695, 0.8798805766838528, 0.8781581611020282, 0.8776155532364472, 0.8798919637300929, 0.8792438003175519, 0.8784528236842492, 0.8815072116243372, 0.8803015988828137, 0.8750046602640532, 0.8813076953963279, 0.8762971658750535, 0.8797480014937074, 0.8805192513492351, 0.883060431341844, 0.8821785902504935, 0.8821094640581664, 0.8756661745348316, 0.8799627701946547, 0.8837654518776159, 0.8618697473588119, 0.8852662259702702, 0.8840898941829228, 0.8838975243993068, 0.8826123566930486, 0.8837599093832157, 0.8834046156186848, 0.8843327981419951, 0.8837321800279645, 0.8852713915512311, 0.8850639814093006, 0.8821484269200793, 0.8847205446533368, 0.8851676003214128, 0.8846126551045654, 0.8855910740757725, 0.8837478156345474, 0.8850045122474332, 0.886595829419009, 0.8867571140140909, 0.8862906743320993, 0.8868487541065547, 0.8861813171527188, 0.8862918889791059, 0.8862789954739048, 0.8874068899888518, 0.887286708999621, 0.8867680544873319, 0.8862455241297572, 0.8874586730915984, 0.885445323257182, 0.8863643319278973, 0.8874518178320181, 0.887343557868164, 0.8869857685817635, 0.8869456049301835, 0.8870900427299823, 0.8869086965292989, 0.8869002953585523, 0.8862831312822962, 0.8853751053608058, 0.8862967131027778, 0.8871426975560056, 0.8879198349856671, 0.8872456814020627, 0.8871534970155857, 0.888303701601884, 0.8876215630871889, 0.8868372330887087, 0.8877398283920016, 0.8887014141502116, 0.8877453220152688, 0.8879609051183841, 0.8882680043132488, 0.8872807575542183, 0.8878170308438057, 0.8882409241634792, 0.8876844788942823, 0.8878083987353321, 0.8885488104081654, 0.8884396786533291, 0.8878818070402005, 0.8875479211340348, 0.8888140028881096, 0.8881210876087788, 0.8876052444934037, 0.8882221912253464, 0.8889159554352961, 0.888151745697175, 0.8886120252974633, 0.8880929268261936]

x_all=list(range(200))
# create four lists of (x, y) pairs using zip()
line1 = list(zip(x_all, Deeplabv3_F1))
line2 = list(zip(x_all, pspnet_f1))
line3 = list(zip(x_all, segnet_F1))
line4 = list(zip(x_all, uner_F1))
plt.figure(figsize=(9, 6))
plt.xlabel('Epoch')
plt.ylabel('F1-Score')
# plot the lines separately with their own settings and label
plt.plot([p[0] for p in line3], [p[1] for p in line3], color='blue', marker='.', linestyle='solid',
         linewidth=1, markersize=1, label='Improved_DeepLabV3+')
plt.plot([p[0] for p in line1], [p[1] for p in line1], color='green', marker='.', linestyle='solid',
         linewidth=1, markersize=1, label='SegNet')
plt.plot([p[0] for p in line2], [p[1] for p in line2], color='red', marker='.', linestyle='solid',
         linewidth=1, markersize=1, label='PSPNet')
plt.plot([p[0] for p in line4], [p[1] for p in line4], color='yellow', marker='.', linestyle='solid',
         linewidth=1, markersize=1, label='UNet')

# add a legend to the plot
plt.legend()
# plt.show()

plt.savefig("F1.png", dpi=500)
plt.clf()    # create four lists of (x, y) pairs using zip()