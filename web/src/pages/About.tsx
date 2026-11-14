import { Link } from 'react-router-dom';
import { ArrowRight, Eye, Scale, Fingerprint } from 'lucide-react';
export default function About({ privacy = false }: { privacy?: boolean }) {
  return (
    <div className="container page prose-page">
      <section className="page-heading">
        <div className="eyebrow">{privacy ? 'DEINE GEDANKEN GEHÖREN DIR' : 'UNSER ANSATZ'}</div>
        <h1>
          {privacy ? (
            <>
              Wenig Daten.
              <br />
              Viel Freiraum.
            </>
          ) : (
            <>
              Mehr verstehen.
              <br />
              <em>Nicht vorsortieren.</em>
            </>
          )}
        </h1>
        <p>
          {privacy
            ? 'WEITBLICK funktioniert ohne Konto, ohne Tracking und ohne politisches Profil.'
            : 'Medien setzen unterschiedliche Schwerpunkte. Wir wollen diese Unterschiede sichtbar machen – ohne Menschen oder Redaktionen in feste Schubladen zu stecken.'}
        </p>
      </section>
      {privacy ? (
        <div className="prose">
          <h2>Was im Browser gespeichert wird</h2>
          <p>
            Deine Merkliste, der Status der Einführung und deine Reflexionen werden im lokalen
            Speicher dieses Browsers abgelegt. Sie werden von der Anwendung nicht an einen Server
            übertragen oder zwischen Geräten synchronisiert. Andere Personen mit Zugriff auf dieses
            Browserprofil können sie möglicherweise lesen.
          </p>
          <h2>Wie du Daten löschst</h2>
          <p>
            Entferne Themen über das Lesezeichen. Reflexionen lassen sich im jeweiligen Thema mit
            „Meine Antworten löschen“ zurücksetzen. Alle gespeicherten Daten kannst du auch über die
            Website-Daten in deinen Browsereinstellungen löschen.
          </p>
          <h2>Beim Laden der Website</h2>
          <p>
            Die Anwendung lädt Code, Illustrationen und JSON-Daten vom jeweiligen Hosting-Server.
            Sie verwendet keine Analyse-, Werbe- oder externen Schrift-Dienste. Der Hosting-Anbieter
            kann technische Zugriffsdaten verarbeiten. Dessen konkrete Angaben müssen vor einem
            öffentlichen Regelbetrieb ergänzt werden.
          </p>
          <h2>Externe Links</h2>
          <p>
            Links zu Medien und GitHub führen auf fremde Websites mit eigenen Datenschutzregeln. Es
            werden keine Artikel oder externen Einbettungen automatisch nachgeladen.
          </p>
          <h2>Projektstatus</h2>
          <p>
            Dies ist ein offener Entwicklungsprototyp. Betreiberkontakt und die für einen
            öffentlichen Betrieb erforderlichen Anbieterinformationen müssen vor dem Start ergänzt
            werden.
          </p>
        </div>
      ) : (
        <>
          <div className="method-pillars">
            <article>
              <Eye />
              <h2>Perspektive ist kein Etikett.</h2>
              <p>
                Ein Medium kann bei verschiedenen Themen unterschiedliche Positionen vertreten.
                Deshalb betrachten wir den einzelnen Beitrag und seinen Kontext.
              </p>
            </article>
            <article>
              <Scale />
              <h2>Vielfalt ist kein Wahrheitsbeweis.</h2>
              <p>
                Nicht jede Behauptung ist gleich gut belegt. Ein fairer Vergleich muss Qualität,
                Unsicherheit und fehlende Belege getrennt benennen.
              </p>
            </article>
            <article>
              <Fingerprint />
              <h2>Du bist kein Meinungsprofil.</h2>
              <p>
                Deine Reflexion ist privat. Wir vergeben keine Punkte für eine politische Richtung
                und drängen dich nicht zu einem bestimmten Ergebnis.
              </p>
            </article>
          </div>
          <div className="prose">
            <h2>Was du hier tatsächlich siehst</h2>
            <p>
              Die Themenübersicht enthält sechs illustrative Szenarien. Ihre Texte wurden für den
              Prototyp verfasst; „Gesellschaftslabor“, „Wirtschaftsfenster“ und „Praxisblick“ sind
              fiktive Quellen. Sie dürfen nicht als Berichterstattung oder als Position realer
              Medien verstanden werden.
            </p>
            <p>
              Daneben steht das echte CSV-Archiv aus dem ursprünglichen Forschungsprojekt. Es
              enthält Schlagzeilen von NZZ, SPIEGEL, taz und FOCUS. Die Einträge wurden lediglich
              normalisiert und pro Quelle dedupliziert. Ein Teil enthält Navigation oder Werbung.
              Artikel-URLs, Autorenschaft, Volltexte und Veröffentlichungsdaten fehlen.
            </p>
            <h2>Was die Farben bedeuten</h2>
            <p>
              <strong>Violett: Gesellschaft & Teilhabe.</strong> Wer ist betroffen, wer wird gehört?
              <br />
              <strong>Ocker: Kosten & Wirtschaft.</strong> Wer trägt Kosten, wer profitiert?
              <br />
              <strong>Türkis: Praxis & Umsetzung.</strong> Was funktioniert unter welchen
              Bedingungen?
            </p>
            <p>
              Diese Schwerpunkte sind nicht exklusiv. Die Demo ordnet jeder Perspektive einen
              Schwerpunkt zu, um die Bedienung zu zeigen. Die Balken zählen diese Zuordnungen; sie
              messen keine politische Tendenz, Glaubwürdigkeit oder repräsentative Medienverteilung.
            </p>
            <h2>Was für echte Einordnungen noch fehlt</h2>
            <ol>
              <li>
                Ein transparenter Katalog erlaubter Datenquellen mit Artikel-URLs, Zeitstempeln,
                Textart und Nutzungsrechten.
              </li>
              <li>
                Nachvollziehbare Gruppierung von Beiträgen zum gleichen Ereignis. Ähnliche Wörter
                allein sind kein ausreichender Beleg.
              </li>
              <li>
                Dokumentierte Einordnung pro Beitrag: Belegstellen, Begründung, Unsicherheit,
                prüfende Person und Revisionsdatum.
              </li>
              <li>
                Ein Verfahren für Korrekturen, Widerspruch und regelmäßige Qualitätskontrolle.
              </li>
            </ol>
            <h2>Vom Forschungsexperiment zur Lesehilfe</h2>
            <p>
              Challenge Your Opinion begann mit dem Sammeln von Schlagzeilen, Wortfrequenzen und
              Topic-Modellen. WEITBLICK ergänzt eine zugängliche Oberfläche. Die vorhandenen
              Analysen bleiben erhalten; ihre Ergebnisse werden nicht als überprüfte politische
              Klassifikation ausgegeben.
            </p>
            <Link className="button" to="/archiv">
              Den Originalkorpus erkunden <ArrowRight size={17} />
            </Link>
          </div>
        </>
      )}
    </div>
  );
}
